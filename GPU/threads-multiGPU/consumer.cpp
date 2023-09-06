#include "consumer.h"
#include <chrono>
#include <iostream>

void cConsumer::operator() () {
    // acquire lock to prevent the producer to mess up
    // with the transfer buffer while the latter is used

    for (int cycle = 0; cycle < 20; cycle++) {
        // at the beginning of each cycle, copy what's in 
        // the transfer buffer; perhaps fresh data is in there
        pTransferBufferCoordination->lock();
        memcpy(outcome, transferBuffer, N_MANUFACTURED_ITEMS * sizeof(int));
        pTransferBufferCoordination->unlock();

        for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
            outcome[j] += this->localUse(j);
        }

        std::cout << "Consumer side values. Cycle: " << cycle << std::endl;
        for (int j = 0; j < N_MANUFACTURED_ITEMS; j++) {
            std::cout << outcome[j] << std::endl;
        }

    }
}

int cConsumer::localUse(int val) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return 2 * val;
}
