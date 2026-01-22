#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "training.h"
#include "chess_board.h"
#include "move_generator.h"
#include "nnue.h"
#include "search.h"
#include "transposition_table.h"
#include "utility.h"

constexpr int MAX_RANDOM_MOVES = 10;

typedef struct TrainingThread {
    SearchThread st;
    pthread_t id;
    uint64_t seed;
    FILE *file;
} TrainingThread;

static TrainingThread tth[32]; // TODO
static TT transpositionTable[32]; // TODO
static atomic_bool stop;
static int activeThreads;

typedef struct GameData {
    struct GameData *prev;
    Score score;
    char fen[128];
} GameData;

// Score parameter is relative to the side to move
static void createGameData(GameData *current, GameData *previous, const ChessBoard *restrict board, Score score) {
    current->prev  = previous;
    current->score = board->sideToMove ? -score : score;
    getFEN(board, current->fen);
}

// Outcome of the game is relative to white (1.0 == white won, 0.5 == draw, 0.0 == black won)
static int writeGameData(const GameData *restrict data, FILE *restrict file, double outcome) {
    int positions = 0;
    // A dummy node is placed at the end, this node can be found by checking if it leads to null in prev
    while (data->prev) {
        positions++;
        fprintf(file, "%s | %d | %.1f\n", data->fen, data->score, outcome);
        data = data->prev;
    }
    return positions;
}

static inline void swap(MoveObject *mo1, MoveObject *mo2) {
    MoveObject temp = *mo1;
    *mo1 = *mo2;
    *mo2 = temp;
}

static inline bool isCheckmate(Score score) {
    return score <= -GUARANTEE_CHECKMATE || score >= GUARANTEE_CHECKMATE;
}

static inline bool isStalemate(Score score, Move bestMove) {
    return score == DRAW && !bestMove;
}

static inline bool isEndOfGame(const ChessBoard *restrict board, const MoveObject *restrict moveObj) {
    return isCheckmate(moveObj->score) || isStalemate(moveObj->score, moveObj->move) || isDraw(board);
}

// Randomly plays the first 5-10 moves
static void playRandomMoves(ChessBoard *restrict board, ChessBoardHistory *restrict history, Accumulator *restrict accumulator, TrainingThread *tt) {
    int numberOfRandomMoves = random64BitNumber(&tt->seed) % 6 + 5;
    for (int i = 0; i < numberOfRandomMoves; i++) {
        MoveObject moveList[MAX_MOVES];
        MoveObject *startList = moveList;
        MoveObject *endList = createMoveList(board, moveList, CAPTURES);
        endList = createMoveList(board, endList, NON_CAPTURES);
        size_t moveListSize = endList - startList;
        while (moveListSize) {
            MoveObject *moveObj = &startList[random64BitNumber(&tt->seed) % moveListSize];
            Move move = moveObj->move;
            if (isLegalMove(board, move)) {
                makeMove(board, &history[i], accumulator, move);
                break;
            }
            moveListSize--;
            swap(startList++, moveObj);
        }
    }
}

static void playGame(TrainingThread *tt, GameData *restrict previous) {
    ChessBoard *board = &tt->st.board;
    Accumulator *accumulator = tt->st.accumulator;
    ChessBoardHistory history;
    GameData current;
    MoveObject *bestMove = startSearch(&tt->st);
    if (!getCheckers(board) && !isCheckmate(bestMove->score) && !insufficientMaterial(board)) { // TODO: What positions to save?
        createGameData(&current, previous, board, bestMove->score);
        previous = &current;
    }
    if (isEndOfGame(board, bestMove)) {
        double outcome = 0.5;
        if (isCheckmate(bestMove->score)) {
            Colour winner = bestMove->score > 0 ? board->sideToMove : !board->sideToMove;
            outcome = winner ? 0.0 : 1.0;
        }
        writeGameData(previous, tt->file, outcome);
        return;
    }
    makeMove(board, &history, accumulator, bestMove->move);
    playGame(tt, previous);
}

static void playRandomGame(TrainingThread *tt) {
    ChessBoard board;
    ChessBoardHistory history[MAX_RANDOM_MOVES + 1];
    Accumulator accumulator;
    GameData dummy = {.prev = nullptr};
    parseFEN(&board, history, &accumulator, START_POS);
    playRandomMoves(&board, &history[1], &accumulator, tt);
    createSearchThread(&tt->st, &board, tt->st.tt, &accumulator, 1000000000 / 8, false);
    playGame(tt, &dummy); // TODO: Is it safe to write data for position that randomly is draw?
}

static void* startTraining(void *trainingThread) {
    TrainingThread *tt = trainingThread;
    while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
        playRandomGame(tt);
        clearTranspositionTable(tt->st.tt);
    }
    return nullptr;
}

static void startTrainingThread(TrainingThread *restrict tthr, uint64_t seed, const char *restrict filename) {
    tthr->seed = seed;
    tthr->file = fopen(filename, "ab+");
    pthread_create(&tthr->id, nullptr, startTraining, tthr);
}

static void stopTrainingThread(TrainingThread *tthr, FILE *restrict merge, int thIndex) {
    pthread_join(tthr->id, nullptr);
    destroyTranspositionTable(tthr->st.tt);
    fflush(tthr->file);
    rewind(tthr->file);

    size_t read;
    char data[2048];
    while ((read = fread(data, sizeof(data[0]), sizeof(data), tthr->file)))
        fwrite(data, sizeof(data[0]), read, merge);

    fclose(tthr->file);
    snprintf(data, sizeof(data), "training_data%02d.txt", thIndex);
    remove(data);
}

void startTrainingThreads(const UCI_Configuration *restrict config) {
    if (activeThreads) stopTrainingThreads();
    activeThreads = config->threads;
    printf("info string training started with %d threads\n", activeThreads);

    atomic_store_explicit(&stop, false, memory_order_relaxed);
    uint64_t seed = time(nullptr);
    char filename[32];
    for (int i = 0; i < activeThreads; i++) {
        uint64_t random = splitMix64(&seed);
        random = random64BitNumber(&random);
        snprintf(filename, sizeof(filename), "training_data%02d.txt", i); // TODO: Make directory
        tth[i].st.tt = &transpositionTable[i];
        createTranspositionTable(&transpositionTable[i], config->hashSize);
        startTrainingThread(&tth[i], random, filename);
    }
}

void stopTrainingThreads() {
    if (!activeThreads) return;
    atomic_store_explicit(&stop, true, memory_order_relaxed);
    FILE *merge = fopen("training_data.txt", "a");
    for (int i = 0; i < activeThreads; i++) {
        printf("info string stopping thread: %d\n", i);
        stopTrainingThread(&tth[i], merge, i);
        printf("info string thread: %d, stopped\n", i);
    }
    fclose(merge);
    activeThreads = 0;
}
