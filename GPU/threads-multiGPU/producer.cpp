#include "producer.h"
#include <chrono>
#include <iostream>

void cProducer::operator() () {
    // perform MANUFACTURE_LOOPS loops, producing stuff in each of them; 
    // once produced, it should be made available to the consumer via memcpy
    for (int i = 0; i < MANUFACTURE_LOOPS; i++) {
        //produce something here
        for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
            product[j] += this->costlyProductionStep(j);
        }

        // acquire lock and supply the consumer
        pTransferBufferCoordination->lock();
        memcpy(pConsumerBuffer, product, N_MANUFACTURED_ITEMS * sizeof(int));
        pTransferBufferCoordination->unlock();
    }
}

int cProducer::costlyProductionStep(int val) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 2 * val + 1;
}