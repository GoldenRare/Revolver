#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "uci.h"
#include "chess_board.h"
#include "move_generator.h"
#include "nnue.h"
#include "transposition_table.h"
#include "utility.h"
#include "search.h"
#include "training.h"

// Official UCI Commands
constexpr char GO          [] = "go"        ;
constexpr char IS_READY    [] = "isready"   ;
constexpr char POSITION    [] = "position"  ;
constexpr char QUIT        [] = "quit"      ;
constexpr char SET_OPTION  [] = "setoption" ;
constexpr char UCI         [] = "uci"       ;
constexpr char UCI_NEW_GAME[] = "ucinewgame";

// Unofficial UCI Commands
constexpr char BENCHMARK[] = "benchmark";
constexpr char EVAL     [] = "eval"     ;
constexpr char FEN      [] = "fen"      ;
constexpr char TRAIN    [] = "train"    ;

static ChessBoardHistory histories[1024]; // TODO: New design? Can use position halfmove clock to determine max size

static void go(UCI_Configuration *restrict config) {
    constexpr char binc [] = "binc" ;
    constexpr char btime[] = "btime";
    constexpr char depth[] = "depth";
    constexpr char winc [] = "winc" ;
    constexpr char wtime[] = "wtime";
    // TODO: Options to potentially implement. All times are in msec
    //constexpr char nodes   [] = "nodes"   ;
    //constexpr char movetime[] = "movetime";
    //constexpr char infinite[] = "infinite";

    uint64_t bIncNs, bTimeNs, wIncNs, wTimeNs, stmSearchTimeNs, searchTimeNs;
    bIncNs = bTimeNs = wIncNs = wTimeNs = stmSearchTimeNs = searchTimeNs = 0;
    char *token;
    while ((token = strtok(nullptr, " ")))
        if      (strcmp(token, binc    ) == 0) bIncNs = strtoull(strtok(nullptr, " "), nullptr, 10) * 1000000;
        else if (strcmp(token, btime   ) == 0) bTimeNs = strtoull(strtok(nullptr, " "), nullptr, 10) * 1000000;
        else if (strcmp(token, depth   ) == 0) strtok(nullptr, " ");
        else if (strcmp(token, winc    ) == 0) wIncNs = strtoull(strtok(nullptr, " "), nullptr, 10) * 1000000;
        else if (strcmp(token, wtime   ) == 0) wTimeNs = strtoull(strtok(nullptr, " "), nullptr, 10) * 1000000;

    stmSearchTimeNs = config->board.sideToMove ? bTimeNs / 20 + bIncNs / 2 : wTimeNs / 20 + wIncNs / 2;
    searchTimeNs = stmSearchTimeNs ? stmSearchTimeNs : 1000000000;
    startSearchThreads(config, searchTimeNs);
}

static void isReady() {
    puts("readyok");
}

static void processMoves(ChessBoard *restrict board, Accumulator *restrict accumulator) {
    char *moveStr;
    int i = 1;
    while ((moveStr = strtok(nullptr, " "))) {
        MoveObject moveList[MAX_MOVES];
        // TODO: Make a legal move generation stage
        MoveObject *endList = createMoveList(board, moveList, CAPTURES);
        endList = createMoveList(board, endList, NON_CAPTURES);
        char moveToName[6];
        for (MoveObject *startList = moveList; startList < endList; startList++) {
            moveToString(moveToName, startList->move);
            if (strcmp(moveStr, moveToName) == 0) {
                makeMove(board, &histories[i++], accumulator, startList->move);
                break;
            }
        }
    }
}

// TODO: Fix setting FEN because of halfmove clock
static void position(ChessBoard *restrict board, Accumulator *restrict accumulator) {
    constexpr char fen[] = "fen";

    const char *fenStr = START_POS;
    if (strcmp(strtok(nullptr, " "), fen) == 0) {
        fenStr = strtok(nullptr, " ");
        for (int i = 0; i < 5; i++) *(strtok(nullptr, " ") - 1) = ' ';
    }

    parseFEN(board, &histories[0], accumulator, fenStr);
    if (strtok(nullptr, " ")) processMoves(board, accumulator); // Assumes token is "moves" if there
}

static void setOption(UCI_Configuration *restrict config) {
    constexpr char Hash   [] = "Hash"   ;
    constexpr char Threads[] = "Threads";

    
    strtok(nullptr, " "); // Discard name string
    char *token = strtok(nullptr, " ");
    strtok(nullptr, " "); // Discard value string

    if      (strcmp(token, Hash   ) == 0) createTranspositionTable(&config->tt, config->hashSize = strtoull(strtok(nullptr, " "), nullptr, 10));
    else if (strcmp(token, Threads) == 0) config->threads = strtoul(strtok(nullptr, " "), nullptr, 10);
}

static void uci() {
    puts("id name Revolver 2.0");
    puts("id author Deshawn Mohan");
    puts("option name Hash type spin default 16 min 1 max 1024"); // TODO: What to make max?
    puts("option name Threads type spin default 1 min 1 max 255");
    puts("uciok");
}

static void uciNewGame(TT *restrict tt) {
    clearTranspositionTable(tt);
}

static uint64_t perft(ChessBoard *restrict board, Depth depth) {
    uint64_t nodes = 0;
    ChessBoardHistory history;
    MoveObject moveList[MAX_MOVES];
    MoveObject *endList = createMoveList(board, moveList, CAPTURES);
    endList = createMoveList(board, endList, NON_CAPTURES);
    if (depth == 1)
        for (MoveObject *moveObj = moveList; moveObj != endList; moveObj++) 
            nodes += isLegalMove(board, moveObj->move);
    else 
        for (MoveObject *moveObj = moveList; moveObj != endList; moveObj++) {
            Move move = moveObj->move;
            if (!isLegalMove(board, move)) continue;
            makeMove(board, &history, nullptr, move);
            nodes += perft(board, depth - 1);
            undoMove(board, move);
        }
    return nodes;
}

static void benchmark() {
    Depth depth = strtoul(strtok(nullptr, " "), nullptr, 10);
    FILE *perftFile = fopen("perft_test_cases.txt", "r");
    char line[256];

    double totalTime = 0.001; // TODO: Non ideal way to guard divide by 0 below
    uint64_t actualNodes = 0, expectedNodes = 0;
    printf("info string benchmark starting, depth: %u\n", depth);
    while (fgets(line, sizeof(line), perftFile)) {
        ChessBoard board = {0};
        ChessBoardHistory history = {0};
        parseFEN(&board, &history, nullptr, strtok(line, ","));

        clock_t start = clock(); // TODO: Consider a more accurate clock
        actualNodes += perft(&board, depth);
        totalTime += (double) (clock() - start) / CLOCKS_PER_SEC;
        
        for (int i = 0; i < depth - 1; i++) strtok(nullptr, ",");
        expectedNodes += strtoull(strtok(nullptr, ","), nullptr, 10);
    }

    printf("info string benchmark %s, expected positions: %llu, positions got: %llu\n", expectedNodes == actualNodes ? "passed" : "failed", expectedNodes, actualNodes);
    printf("info string total time: %.2lf sec, positions/sec: %.0lf\n", totalTime, actualNodes / totalTime);
    fclose(perftFile);
}

static void eval(const Accumulator *restrict accumulator, Colour stm) {
    printf("Static Evaluation: %d\n", evaluation(accumulator, stm));
}

static void fen(const ChessBoard *restrict board) {
    char fen[128];
    getFEN(board, fen);
    puts(fen);
}

static void train(const UCI_Configuration *restrict config) {
    startTrainingThreads(config);
}

void uciLoop() {
    // Default configuration
    UCI_Configuration config = {.hashSize = 16, .threads = 1};
    parseFEN(&config.board, &histories[0], &config.accumulator, START_POS);
    createTranspositionTable(&config.tt, config.hashSize);

    char input[4096]; // Assumes input is large enough to hold '\n' from stdin
    char *token = nullptr;
    setvbuf(stdout, nullptr, _IONBF, 0);
    while (!token || strcmp(token, QUIT) != 0) {
        fgets(input, sizeof(input), stdin);
        input[strlen(input) - 1] = '\0'; // TODO: Can remove the length check
        token = strtok(input, " ");
        if (!token) continue;

        // Official UCI Commands
        if      (strcmp(token, GO          ) == 0) go(&config);
        else if (strcmp(token, IS_READY    ) == 0) isReady();
        else if (strcmp(token, POSITION    ) == 0) position(&config.board, &config.accumulator);
        else if (strcmp(token, SET_OPTION  ) == 0) setOption(&config);
        else if (strcmp(token, UCI         ) == 0) uci();
        else if (strcmp(token, UCI_NEW_GAME) == 0) uciNewGame(&config.tt);

        // Unofficial UCI Commands
        else if (strcmp(token, BENCHMARK) == 0) benchmark();
        else if (strcmp(token, EVAL     ) == 0) eval(&config.accumulator, config.board.sideToMove);
        else if (strcmp(token, FEN      ) == 0) fen(&config.board);
        else if (strcmp(token, TRAIN    ) == 0) train(&config);
    }
    stopTrainingThreads();
}
