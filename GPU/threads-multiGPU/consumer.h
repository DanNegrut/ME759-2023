#pragma once
#include <mutex>
#include "problemParams.h"

class cConsumer {
private:
    std::mutex* pTransferBufferCoordination;
    int transferBuffer[N_MANUFACTURED_ITEMS] = { 0, 0, 0 };
    int outcome[N_MANUFACTURED_ITEMS];

    int localUse(int val);

public:
    cConsumer(std::mutex* pTBC) :pTransferBufferCoordination(pTBC) {}
    ~cConsumer() {}
    int* pDesitnationBuffer() { return transferBuffer; }
    void operator() ();
};