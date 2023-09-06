#ifndef KERN_TEST_H
#define KERN_TEST_H

#define IMPORT_KERNEL(k) extern "C" void* kern_##k
#define KERNEL_HANDLE(k) (kern_##k)

IMPORT_KERNEL(add_vector);

#endif // KERN_TEST_H