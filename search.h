#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>
#include <time.h>
#include "chess_board.h"
#include "nnue.h"
#include "transposition_table.h"
#include "uci.h"
#include "utility.h"

typedef struct SearchThread {
    Accumulator accumulator[512]; // TODO: Where to store accumulator and sizing. Struct alignment?
    ChessBoard board;
    TT *tt;
    uint64_t startNs; // TODO: Could change implementation
    uint64_t maxSearchTimeNs;
    uint64_t nodes;
    MoveObject bestMove;
    uint8_t ply;
    bool print;
    bool stop;
} SearchThread;

static inline uint64_t getTimeNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

static inline void createSearchThread(SearchThread *st, const ChessBoard *restrict board, TT *tt, Accumulator *accumulator, uint64_t maxSearchTimeNs, bool print) {
    st->board = *board; // TODO: The history pointer is a shallow copy, consider using a deep copy
    st->tt = tt;
    st->accumulator[0] = *accumulator;
    st->maxSearchTimeNs = maxSearchTimeNs;
    st->nodes = 0;
    st->ply = 0;
    st->print = print;
    st->stop = false;
}

static inline bool outOfTime(SearchThread *st) {
    return st->stop = getTimeNs() - st->startNs >= st->maxSearchTimeNs;
}

void* startSearch(void *searchThread);
void startSearchThreads(UCI_Configuration *restrict config, uint64_t searchTimeNs);

#endif
