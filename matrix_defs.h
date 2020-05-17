
#ifndef MATRIX_DEFS_H
#define MATRIX_DEFS_H

typedef double floating_type;

// Macros for handling matricies.
// These macros manipulate a linear array as if it was a two dimensional array.
// TODO: Create a matrix abstraction? Or would the overhead of doing so be too great?
#define MATRIX_MAKE( size )  ((floating_type *)malloc( (size) * (size) * sizeof( floating_type ) ))
#define MATRIX_DESTROY( matrix )                       ( free( matrix ) )
#define MATRIX_GET( matrix, size, row, column )        ( (matrix)[(row)*(size) + (column)] )
#define MATRIX_GET_REF( matrix, size, row, column )    (&(matrix)[(row)*(size) + (column)] )
#define MATRIX_GET_ROW( matrix, size, row )            (&(matrix)[(row)*(size)] )
#define MATRIX_PUT( matrix, size, row, column, value ) ( (matrix)[(row)*(size) + (column)] = (value) )

#endif
