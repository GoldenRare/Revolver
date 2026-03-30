#include <immintrin.h>
#include <stdint.h>
#include "nnue.h"
#include "utility.h"

constexpr int FLIP_MASK       = 0b111000;
constexpr int SCORE_SCALE     =      400;
constexpr int QUANTIZATION_A  =      255;
constexpr int QUANTIZATION_B  =       64;
constexpr int PERSPECTIVE     =        2;

typedef struct Network {
    int16_t accumulatorWeights[COLOURS][PIECE_TYPES - 1][SQUARES][LAYER1];
    int16_t accumulatorBiases[LAYER1];

    int16_t outputWeights[LAYER1 * PERSPECTIVE];
    int16_t outputBias;
} Network;

static const uint8_t networkData[] = {
    #embed "nnue.bin"
};

static const Network *network = (const Network *) networkData;

void accumulatorReset(Accumulator *restrict accumulator) {
    for (int i = 0; i < LAYER1; i++) {
        accumulator->accumulator[WHITE][i] = network->accumulatorBiases[i];
        accumulator->accumulator[BLACK][i] = network->accumulatorBiases[i];
    }
}

void accumulatorAdd(Accumulator *restrict accumulator, Colour c, PieceType pt, Square sq) {
    pt--;
    for (int i = 0; i < LAYER1; i++) accumulator->accumulator[WHITE][i] += network->accumulatorWeights[c    ][pt][sq ^ FLIP_MASK][i];
    for (int i = 0; i < LAYER1; i++) accumulator->accumulator[BLACK][i] += network->accumulatorWeights[c ^ 1][pt][sq            ][i];
}

void accumulatorSub(Accumulator *restrict accumulator, Colour c, PieceType pt, Square sq) {
    pt--;
    for (int i = 0; i < LAYER1; i++) accumulator->accumulator[WHITE][i] -= network->accumulatorWeights[c    ][pt][sq ^ FLIP_MASK][i];
    for (int i = 0; i < LAYER1; i++) accumulator->accumulator[BLACK][i] -= network->accumulatorWeights[c ^ 1][pt][sq            ][i];
}

void accumulatorAddSub(Accumulator *restrict accumulator, Colour c, PieceType pt, Square fromSquare, Square toSquare) {
    pt--;
    for (int i = 0; i < LAYER1; i++)
        accumulator->accumulator[WHITE][i] += network->accumulatorWeights[c][pt][toSquare ^ FLIP_MASK][i] - network->accumulatorWeights[c][pt][fromSquare ^ FLIP_MASK][i];

    for (int i = 0; i < LAYER1; i++)
        accumulator->accumulator[BLACK][i] += network->accumulatorWeights[c ^ 1][pt][toSquare][i] - network->accumulatorWeights[c ^ 1][pt][fromSquare][i];
}

void accumulatorAddSubPromotion(Accumulator *restrict accumulator, Colour c, PieceType pt, Square fromSquare, Square toSquare) {
    pt--;
    for (int i = 0; i < LAYER1; i++)
        accumulator->accumulator[WHITE][i] += network->accumulatorWeights[c][pt][toSquare ^ FLIP_MASK][i] - network->accumulatorWeights[c][PAWN - 1][fromSquare ^ FLIP_MASK][i];

    for (int i = 0; i < LAYER1; i++)
        accumulator->accumulator[BLACK][i] += network->accumulatorWeights[c ^ 1][pt][toSquare][i] - network->accumulatorWeights[c ^ 1][PAWN - 1][fromSquare][i];
}

// TODO: Make the accumulator aligned
// TODO: Should I have two separate sums, then combine into one sum?
// TODO: Find better intrinsics to use instead of storeu for sumArr
Score evaluation(const Accumulator *restrict accumulator, Colour stm) {
    constexpr int NUMBER_OF_VECTORS = LAYER1 / 16;
    const __m256i zeroVector    = _mm256_setzero_si256();
    const __m256i qaVector      = _mm256_set1_epi16(QUANTIZATION_A);
    const __m256i *stmAcc       = (const __m256i *) accumulator->accumulator[stm    ];
    const __m256i *enemyAcc     = (const __m256i *) accumulator->accumulator[stm ^ 1];
    const __m256i *stmWeights   = (const __m256i *) network->outputWeights;
    const __m256i *enemyWeights = stmWeights + NUMBER_OF_VECTORS;

    __m256i sum = zeroVector;
    for (int i = 0; i < NUMBER_OF_VECTORS; i++) {
        __m256i stmAccVector      = _mm256_loadu_si256(&stmAcc[i]);
        __m256i enemyAccVector    = _mm256_loadu_si256(&enemyAcc[i]);
        __m256i stmWeightVector   = _mm256_load_si256(&stmWeights[i]);
        __m256i enemyWeightVector = _mm256_load_si256(&enemyWeights[i]);

        __m256i stmClamped   = _mm256_min_epi16(_mm256_max_epi16(stmAccVector  , zeroVector), qaVector);
        __m256i enemyClamped = _mm256_min_epi16(_mm256_max_epi16(enemyAccVector, zeroVector), qaVector);

        __m256i stmResult   = _mm256_madd_epi16(_mm256_mullo_epi16(stmClamped  , stmWeightVector  ), stmClamped  );
        __m256i enemyResult = _mm256_madd_epi16(_mm256_mullo_epi16(enemyClamped, enemyWeightVector), enemyClamped);

        sum = _mm256_add_epi32(sum, stmResult  );
        sum = _mm256_add_epi32(sum, enemyResult);
    }

    int32_t sumArr[8];
    _mm256_storeu_si256((__m256i *) sumArr, sum);

    Score score = 0;
    for (int i = 0; i < 8; i++) score += sumArr[i];

    score /= QUANTIZATION_A;
    score += network->outputBias;
    return score * SCORE_SCALE / (QUANTIZATION_A * QUANTIZATION_B);
}
