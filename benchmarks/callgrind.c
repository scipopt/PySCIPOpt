#include <valgrind/callgrind.h>

void bench_start(void)
{
   CALLGRIND_START_INSTRUMENTATION;
   CALLGRIND_ZERO_STATS;
}

void bench_stop(void)
{
   CALLGRIND_STOP_INSTRUMENTATION;
   CALLGRIND_DUMP_STATS;
}
