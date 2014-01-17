#ifndef LOWLTYPES_H
#define LOWLTYPES_H

#include <limits.h>

/* define some basic error codes. */
extern const int LOWLERR_NOTANERROR_ACTUALLYHUGESUCCESS_CONGRATS;
extern const int LOWLERR_BADMALLOC;
extern const int LOWLERR_BADINPUT;
extern const int LOWLERR_INDEXOUTOFRANGE;

/* define basic types used throughout lowl. */

typedef int lowl_bool;
#define TRUE 1
#define FALSE 0

/* lowl_keys are used as basic identifiers. We hash on these. */
typedef unsigned long lowl_key;
#define LOWL_KEY_MIN 0
#define LOWL_KEY_MAX ULONG_MAX

/* lowl_hashoutputs are the outputs of hash functions. */
typedef unsigned int lowl_hashoutput;
#define LOWL_HASHOUTPUT_MIN 0
#define LOWL_HASHOUTPUT_MAX UINT_MAX

/* lowl_counts are the type used for storing counts of objects in
	algorithms such as count-min sketch and similar.	*/
typedef unsigned int lowl_count;
#define LOWL_COUNT_MIN 0
#define LOWL_COUNT_MAX UINT_MAX

#endif
