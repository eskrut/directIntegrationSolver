#ifndef PREPAREMESH_H
#define PREPAREMESH_H

class sbfMesh;
class sbfStiffMatrixBlock3x3;

//! Allocates memory and creates mesh in respect to discretisation parameter
sbfMesh * createMesh(int discretParam = 10);

//! Allocates memory and creates stiffness matrix
sbfStiffMatrixBlock3x3 * createStiffMatrix(sbfMesh *mesh, bool recompute);

#endif // PREPAREMESH_H
