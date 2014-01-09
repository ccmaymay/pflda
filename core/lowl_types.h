#ifndef LOWLTYPES_H
#define LOWLTYPES_H

/* define basic types used throughout lowl. */

/* lowl_keys are used as basic identifiers. We hash on these. */
typedef unsigned long lowl_key;

/* lowl_hashoutputs are the outputs of hash functions. */
typedef unsigned int lowl_hashoutput;

/* lowl_counts are the type used for storing counts of objects in
	algorithms such as count-min sketch and similar.	*/
typedef unsigned int lowl_count;

#endif
