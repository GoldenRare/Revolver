#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include "search.h"
#include "chess_board.h"
#include "utility.h"
#include "transposition_table.h"
#include "move_selector.h"
#include "nnue.h"

constexpr Depth MAX_DEPTH = 255;

typedef enum Node {
    ROOT, PV, NON_PV
} Node;

typedef struct SearchHelper {
    Move pv[MAX_DEPTH]; // TODO: Is it worth saving space by making triangular?
} SearchHelper;

static inline void updatePV(Move move, Move *restrict currentPV, const Move *restrict childrenPV) {
    *currentPV++ = move;
    while((*currentPV++ = *childrenPV++));
}

// TODO: Is this the ideal implementation?
// Always assumes there is at least one move in the principal variation
static inline void pvToString(char *restrict pvStr, char *restrict bestMove, char *restrict ponderMove, const Move *restrict pv) {
    moveToString(bestMove, pv[0]);
    pvStr += moveToString(pvStr, pv[0]);
    *ponderMove = '\0';
    for (Depth depth = 1; depth < MAX_DEPTH && pv[depth]; depth++) {
        if (depth == 1) moveToString(ponderMove, pv[depth]);
        *pvStr++ = ' ';
        pvStr += moveToString(pvStr, pv[depth]);
    }
}

// A move is considered interesting if it is a capture move or a Queen promotion
static inline bool isInteresting(const ChessBoard *restrict board, Move move) {
    return board->pieceTypes[getToSquare(move)] || getMoveType(move) == EN_PASSANT || getMoveType(move) == QUEEN_PROMOTION;
}

// TODO: Ensure our static evaluation after scaled cannot return a false checkmate
static inline Score getReverseFutilityPruningScore(Score staticEvaluation, Depth depth) {
    return staticEvaluation + 150 * depth;
}

// TODO: Ensure our static evaluation after scaled cannot return a false checkmate
static inline Score getRFPMargin(Depth depth) {
    return 150 * depth;
}

// TODO: Should eventually include seldepth
static inline void printSearch(Depth depth, Score score, const char *restrict pvString, const SearchThread *st) {
    uint64_t time = (getTimeNs() - st->startNs) / 1000000;
    uint64_t nps = st->nodes * 1000 / (time + 1);
    char *scoreType = score >= GUARANTEE_CHECKMATE || score <= -GUARANTEE_CHECKMATE ? "mate" : "cp";
    score = score >=  GUARANTEE_CHECKMATE ? ( CHECKMATE - score + 1) / 2
          : score <= -GUARANTEE_CHECKMATE ? (-CHECKMATE - score    ) / 2
          : score;
    printf("info depth %d score %s %d nodes %llu nps %llu time %llu pv %s\n", depth, scoreType, score, st->nodes, nps, time, pvString);
}

static Score quiescenceSearch(Score alpha, Score beta, SearchHelper *restrict sh, SearchThread *st) {
    ChessBoard *board = &st->board;
    st->nodes++;

    /* 1) Draw Detection */
    if (isDraw(board)) return DRAW;
    /*                   */
    
    bool checkers = getCheckers(board);
    const Accumulator *currentAccumulator = &st->accumulator[st->ply    ];
    Accumulator       *childAccumulator   = &st->accumulator[st->ply + 1];
    /* Stand Pat */
    Score bestScore = checkers ? -CHECKMATE + st->ply : evaluation(currentAccumulator, board->sideToMove); // TODO: Could be evaluating a stalemate
    if (bestScore > alpha) {
        if (bestScore >= beta) return bestScore; 
        alpha = bestScore;
    }
    /*           */

    /* Main Moves Loop */
    ChessBoardHistory history;
    MoveSelector ms;
    MoveSelectorState state = checkers ? TT_MOVE : GET_NON_CAPTURE_MOVES; // TODO: Cleanup naming
    createMoveSelector(&ms, board, state, NO_MOVE);

    Move move;
    while ((move = getNextBestMove(board, &ms))) {
        if (!isLegalMove(board, move)) continue;
        
        st->ply++;
        *childAccumulator = *currentAccumulator;
        makeMove(board, &history, childAccumulator, move);
        Score score = -quiescenceSearch(-beta, -alpha, sh, st);
        undoMove(board, move);
        st->ply--;
        
        if (score > bestScore) {
            if (score > alpha) {
                if (score >= beta) return score;
                alpha = score; 
            }
            bestScore = score;
        }
    }
    /*                 */
    return bestScore;
}

static Score alphaBeta(Score alpha, Score beta, Depth depth, Node node, SearchHelper *restrict sh, SearchThread *st) {
    sh->pv[0] = NO_MOVE;

    /* 1) Quiescence Search */
    if (!depth) return quiescenceSearch(alpha, beta, sh, st);
    /*                      */
    
    ChessBoard *board = &st->board;
    st->nodes++;
    /* 2) Draw Detection */
    if ((node != ROOT && isDraw(board)) || outOfTime(st)) return DRAW;
    /*                   */

    /* 3) Transposition Table */
    const bool isPvNode = node != NON_PV;
    bool hasEvaluation;
    Key positionKey = getPositionKey(board);
    PositionEvaluation *pe = probeTranspositionTable(st->tt, positionKey, &hasEvaluation);
    Move ttMove = NO_MOVE;
    if (hasEvaluation) {
        if (!isPvNode && pe->depth >= depth) {
            Bound bound = getBound(pe);
            Score nodeScore = adjustNodeScoreFromTT(pe->nodeScore, st->ply);
            if (bound == EXACT || (bound == LOWER ? nodeScore >= beta : nodeScore <= alpha)) return nodeScore;
        }
        ttMove = pe->bestMove;
    }
    /*                        */

    ChessBoardHistory history;
    SearchHelper *child = sh + 1;
    const Accumulator *currentAccumulator = &st->accumulator[st->ply    ];
    Accumulator       *childAccumulator   = &st->accumulator[st->ply + 1];

    bool checkers = getCheckers(board);
    Score staticEvaluation = checkers ? -INFINITE 
                           : hasEvaluation ? pe->staticEvaluation
                           : evaluation(currentAccumulator, board->sideToMove);
    /** 4) Null Move Pruning **/
    if (!isPvNode && !checkers && depth > 3 && staticEvaluation >= beta && hasNonPawnMaterial(board, board->sideToMove)) {
        st->ply++;
        *childAccumulator = *currentAccumulator;
        makeNullMove(board, &history);
        Score score = -alphaBeta(-beta, -beta + 1, depth - 4, NON_PV, child, st);
        undoNullMove(board);
        st->ply--;
        if (score >= beta) return score;
    }
    /**                      **/

    /** 5) Reverse Futility Pruning **/
    if (!isPvNode && !checkers && staticEvaluation - getRFPMargin(depth) >= beta) return staticEvaluation;
    /**                             **/

    MoveSelector ms;
    createMoveSelector(&ms, board, TT_MOVE, ttMove);

    int legalMoves = 0;
    Score bestScore = -INFINITE, oldAlpha = alpha;
    Move  bestMove  =   NO_MOVE, move;

    /* 6) Move Ordering */
    while ((move = getNextBestMove(board, &ms))) {
        if (!isLegalMove(board, move)) continue;
        legalMoves++;

        bool expectedNonPvNode = !isPvNode || legalMoves > 1;
        /** 7) Futility Pruning **/
        if (expectedNonPvNode && depth < 4 && !checkers && !isInteresting(board, move) && getReverseFutilityPruningScore(staticEvaluation, depth) <= alpha) continue;
        /**                     **/

        /** 8) Late Move Reductions **/
        int reductions = legalMoves > 1 && depth > 1 ? 2 : 1;
        /**                         **/

        st->ply++;
        *childAccumulator = *currentAccumulator;
        makeMove(board, &history, childAccumulator, move);

        /* 9) Principal Variation Search */
        Score score;
        if (expectedNonPvNode) score = -alphaBeta(-alpha - 1, -alpha, depth - reductions, NON_PV, child, st);
        if (isPvNode && (legalMoves == 1 || score > alpha)) score = -alphaBeta(-beta, -alpha, depth - 1, PV, child, st);
        /*                               */
        
        undoMove(board, move);
        st->ply--;

        if (score > bestScore) {
            if (score > alpha) {
                if (score >= beta) {
                    if (!st->stop) savePositionEvaluation(st->tt, pe, positionKey, move, depth, LOWER, adjustNodeScoreToTT(score, st->ply), staticEvaluation);
                    return score;
                }
                updatePV(move, sh->pv, child->pv); // TODO: Only needs to be done once on the last score > alpha, but integrity is lost
                alpha = score; 
            }
            bestScore = score;
            bestMove = move;
        }
    }
    /*                  */

    /* 10) Checkmate and Stalemate Detection */
    if (!legalMoves) bestScore = checkers ? -CHECKMATE + st->ply : DRAW; // TODO: Should this be considered EXACT bound?
    /*                                       */

    if (!st->stop) savePositionEvaluation(st->tt, pe, positionKey, bestMove, depth, bestScore > oldAlpha ? EXACT : UPPER, adjustNodeScoreToTT(bestScore == -INFINITE ? staticEvaluation : bestScore, st->ply), staticEvaluation);
    return bestScore;
}

void* startSearch(void *searchThread) {
    constexpr Score ASPIRATION_WINDOW = 25;
    SearchThread *st = searchThread;
    SearchHelper sh[MAX_DEPTH + 1];
    
    char pvString[2048], bestMove[6], ponderMove[6];
    Score score, alpha = -INFINITE, beta = INFINITE;
    st->startNs = getTimeNs();
    for (Depth depth = 1; depth && !outOfTime(st); depth++) {
        score = alphaBeta(alpha, beta, depth, ROOT, sh, st);
        if (score > alpha && score < beta && !st->stop) {
            alpha = score - ASPIRATION_WINDOW;
            beta = score + ASPIRATION_WINDOW;
            
            st->bestMove.move  = sh[0].pv[0];
            st->bestMove.score = score;

            pvToString(pvString, bestMove, ponderMove, sh[0].pv);
            if (st->print) printSearch(depth, score, pvString, st);
        } else {
            depth--;
            alpha = score > alpha ? alpha : -INFINITE;
            beta  = score < beta  ? beta  :  INFINITE;
        }
    }
    if (st->print) {
        if (ponderMove[0]) printf("bestmove %s ponder %s\n", bestMove, ponderMove);
        else printf("bestmove %s\n", bestMove);
    }
    return &st->bestMove;
}

void startSearchThreads(UCI_Configuration *restrict config, uint64_t searchTimeNs) {
    config->tt.age++;

    pthread_t th;
    SearchThread st;
    createSearchThread(&st, &config->board, &config->tt, &config->accumulator, searchTimeNs, true);
    pthread_create(&th, nullptr, startSearch, &st);
    pthread_join(th, nullptr);
}
