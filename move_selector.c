#include <stdint.h>
#include "move_selector.h"
#include "move_generator.h"
#include "utility.h"

constexpr Score PIECE_VALUE[PIECE_TYPES] = {0, 100, 300, 306, 500, 900, 0};

static void scoreMoves(const ChessBoard *restrict board, MoveSelector *restrict ms) {
    MoveObject *startList = ms->startList;
    while (startList < ms->endList) {
        PieceType capturedPiece = board->pieceTypes[getToSquare(startList->move)];
        if (capturedPiece) {
            startList->score = PIECE_VALUE[capturedPiece] - board->pieceTypes[getFromSquare(startList->move)]; // MVV/LVA
        } else if (getMoveType(startList->move) & EN_PASSANT) {
            startList->score = 90;
        } else {
            startList->score = 0;
        }
        startList++;
    }
}

static Move getNextHighestScoringMove(MoveSelector *restrict ms) {
    MoveObject *highestScoreMove = nullptr;
    int16_t bestScore = -1; // TODO: Score must be the lowest possible
    for (MoveObject *moveObj = ms->startList; moveObj < ms->endList; moveObj++)
        if (moveObj->move != ms->ttMove && moveObj->score > bestScore) {
            bestScore = moveObj->score;
            highestScoreMove = moveObj;
        }
    if (!highestScoreMove) return NO_MOVE;
    Move bestMove = highestScoreMove->move;
    *highestScoreMove = *ms->startList++;
    return bestMove;
}

Move getNextBestMove(const ChessBoard *restrict board, MoveSelector *restrict ms) {
    while (true) {
        switch (ms->state) {
            case TT_MOVE:
                ms->state++;
                return ms->ttMove;
            case Q_SEARCH_CAPTURE_MOVES:
            case CAPTURE_MOVES:
                ms->state++;
                ms->endList = createMoveList(board, ms->moveList, CAPTURES);
                scoreMoves(board, ms);
                break;
            case Q_SEARCH_GET_CAPTURES:
            case GET_CAPTURES:
                Move m;
                if ((m = getNextHighestScoringMove(ms))) return m;
                ms->state++;
                break;
            case NON_CAPTURE_MOVES:
                ms->state++;
                ms->startList = ms->endList;
                ms->endList = createMoveList(board, ms->endList, NON_CAPTURES);
                scoreMoves(board, ms);
                break;
            case GET_NON_CAPTURE_MOVES:
                return getNextHighestScoringMove(ms);
            default:
                return NO_MOVE;
        }
    }
}
