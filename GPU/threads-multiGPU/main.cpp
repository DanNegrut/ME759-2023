#include <iostream>
#include <thread>
#include <mutex>

#include "producer.h"
#include "consumer.h"

int main()
{
    std::mutex transferBufCoordination;
    cConsumer consGuy(&transferBufCoordination);
    cProducer prodGuy(&transferBufCoordination);

    int* pBuffer = consGuy.pDesitnationBuffer();
    prodGuy.setDestinationBuffer(pBuffer);

    std::thread prodThread(std::ref(prodGuy));
    std::thread consThread(std::ref(consGuy));

    consThread.join();
    prodThread.join();
    return 0;
}