#pragma once
#include <mutex>
#include "problemParams.h"

class cProducer {
private:
    std::mutex* pTransferBufferCoordination;
    int* pConsumerBuffer; // this is where the consumer stores data that needs to be produced herein
    int product[N_MANUFACTURED_ITEMS] = { 1,2,3 };

    int costlyProductionStep(int) const;

public:
    cProducer(std::mutex* pTBC) :pTransferBufferCoordination(pTBC) {}
    ~cProducer() {}

    void setDestinationBuffer(int* pC) { pConsumerBuffer = pC; }
    void operator() ();
};