#include <iostream>
#include <memory>
#include "sbf.h"
#include "sbfMatrixIterator.h"
#include "sbfStiffMatrixBlock3x3Iterator.h"
#include "prepareMesh.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <omp.h>

void computeMassDemp(NodesData<double, 3> &mass, NodesData<double, 3> &demp, sbfMesh *mesh, sbfPropertiesSet *propSet, double ksi, bool recalculate);
void getDispl(const double t, double * displ, double ampX, double freqX, double transT);

int main(int argc, char ** argv)
{
    int discretParam, numOutput;
    double t, tStop, dt, dtOut, tNextOut, transT;
    double ampX, freqX, ksi;
    bool recreateMesh = false;
    bool makeNodesRenumbering = true;
    po::options_description desc("Program options");
    desc.add_options()
    ("help,h", "print help message")
    ("dt", po::value<double>(&dt)->default_value(1e-5), "integration time step")
    ("tStop,t", po::value<double>(&tStop)->default_value(1e1), "time range")
    ("transT", po::value<double>(&transT)->default_value(1e-3), "load transition time")
    ("ampX,a", po::value<double>(&ampX)->default_value(1e-3), "displacement amplitude")
    ("freqX,f", po::value<double>(&freqX)->default_value(1e0), "displacement frequency [Hz]")
    ("ksi,k", po::value<double>(&ksi)->default_value(1e-1), "damping factor")
    ("numOutput,n", po::value<int>(&numOutput)->default_value(500), "number of outputs")
    ("discretParam,d", po::value<int>(&discretParam)->default_value(2), "discretisation parameter")
    ("recreateMesh,m", "recreate mesh")
    ("no-nr", "do not optimize nodes numbering")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") || vm.count("h")) { std::cout << desc << "\n"; return 1; }
    if (vm.count("recreateMesh") || vm.count("m")) { recreateMesh = true; }
    if (vm.count("no-nr")) { makeNodesRenumbering = false; }

    t = tNextOut = 0;
    dtOut = tStop/numOutput;
    double a0, a1, a2;
    a0 = 1.0/(dt*dt); a1 = 1.0/(2.0*dt); a2 = 2.0*a0;

    // Create appropriate mesh
    std::unique_ptr<sbfMesh> meshPtr(new sbfMesh);
    if( recreateMesh || meshPtr->readMeshFromFiles() ){
        meshPtr.reset(createMesh(discretParam));
        if ( makeNodesRenumbering ) meshPtr->optimizeNodesNumbering();
        meshPtr->writeMeshToFiles();
    }
    sbfMesh *mesh = meshPtr.get();
    const int numNodes = mesh->numNodes();

    // Create and compute stiffness matrix
    std::unique_ptr<sbfStiffMatrixBlock3x3> stiffPtr(createStiffMatrix(mesh, recreateMesh));
    sbfStiffMatrixBlock3x3 *stiff = stiffPtr.get();

    // Create nodes data storages
    NodesData<double, 3> displ_m1(mesh), displ("displ", mesh), displ_p1(mesh);
    NodesData<double, 3> force(mesh), mass("mass", mesh), demp(mesh), rezKU(mesh);
    mass.null(); force.null(); demp.null(); displ_m1.null(); displ.null();

    // Fill mass and demping arrays
    computeMassDemp(mass, demp, mesh, stiff->propSet(), ksi, recreateMesh);

    // Get indexes of kinematic loaded nodes
    std::vector<int> loadedNodesDown, loadedNodesUp;
    double maxY = mesh->maxY();
    for(int ct = 0; ct < numNodes; ct++){
        if( std::fabs(mesh->node(ct).y()) < 1e-5) loadedNodesDown.push_back(ct);
        if( std::fabs(mesh->node(ct).y() - maxY) < 1e-5 ) loadedNodesUp.push_back(ct);
    }
    const int numLDown = loadedNodesDown.size(), numLUp = loadedNodesUp.size();


    // Time integration loop
    report.createNewProgress("Time integration");
    double tmp[3];
#pragma omp parallel
    {
        // Stiffness matrix iteration halper
        std::unique_ptr<sbfMatrixIterator> iteratorPtr(stiff->createIterator());
        sbfMatrixIterator *iterator = iteratorPtr.get();
        while( t < tStop ) { // Time loop
            // Update displacements of kinematically loaded nodes
#pragma omp master
            getDispl(t, tmp, ampX, freqX, transT);
#pragma omp for nowait
            for(int nodeCt = 0; nodeCt < numLDown; ++nodeCt) for(int ct = 0; ct < 3; ct++)
                displ.data(loadedNodesDown[nodeCt], ct) = tmp[ct];
#pragma omp for
            for(int nodeCt = 0; nodeCt < numLUp; ++nodeCt) for(int ct = 0; ct < 3; ct++)
                displ.data(loadedNodesUp[nodeCt], ct) = tmp[ct]/3;
#pragma omp for
            for(int nodeCt = 0; nodeCt < numNodes; nodeCt++) {
                // Make multiplication of stiffness matrix over displacement vector
                iterator->setToRow(nodeCt);
                double *rez = rezKU.data() + nodeCt*3;
                rez[0] = rez[1] = rez[2] = 0.0;
                while(iterator->isValid()) {
                    int columnID = iterator->column();
                    double *vectPart = displ.data() + columnID*3;
                    double *block = iterator->data();
                    for(int rowCt = 0; rowCt < 3; ++rowCt)
                        for(int colCt = 0; colCt < 3; ++colCt)
                            rez[rowCt] += block[rowCt*3+colCt]*vectPart[colCt];
                    iterator->next();
                }
                // Perform finite difference step
                for(int ct = 0; ct < 3; ct++) {
                    double temp = force.data()[3*nodeCt+ct] - rezKU.data()[3*nodeCt+ct] +
                            a2*mass.data()[3*nodeCt+ct]*displ.data()[3*nodeCt+ct] -
                            (a0*mass.data()[3*nodeCt+ct] - a1*demp.data()[3*nodeCt+ct])
                            *displ_m1.data()[3*nodeCt+ct];
                    displ_p1.data()[3*nodeCt+ct] = temp/(a0*mass.data()[3*nodeCt+ct] + a1*demp.data()[3*nodeCt+ct]);
                }
            }//End of #pragma omp for
#pragma omp master
            {
                // Make output if required
                if(t >= tNextOut) { displ.writeToFile();tNextOut += dtOut;report.updateProgress(t/tStop*100);}
                // Prepere for next step
                t += dt;
            }//End of #pragma omp master
#pragma omp for
            for(int nodeCt = 0; nodeCt < numNodes; nodeCt++) {
                // Perform finite difference step
                for(int ct = 0; ct < 3; ct++) {
                    displ_m1.data()[3*nodeCt+ct] = displ.data()[3*nodeCt+ct];
                    displ.data()[3*nodeCt+ct] = displ_p1.data()[3*nodeCt+ct];
                }
            }//End of #pragma omp for
        } // End time loop
    }//End of #pragma omp parallel
    report.finalizeProgress();

    return 0;
} //End main

void computeMassDemp(NodesData<double, 3> &mass, NodesData<double, 3> &demp, sbfMesh *mesh, sbfPropertiesSet *propSet, double ksi, bool recalculate)
{
    if( recalculate || mass.readFromFile<double>() ) {
        report("Recalculate mass");
        sbfElement *elem;
        sbfElemStiffMatrixHexa8 *stiffHexa8 = new sbfElemStiffMatrixHexa8();
        stiffHexa8->setPropSet(propSet);
        std::vector<int> elemNodeIndexes;
        for(int ctElem = 0; ctElem < mesh->numElements(); ctElem++){//Loop on elements
            elem = mesh->elemPtr(ctElem);
            stiffHexa8->setElem(elem);
            double elemMass = stiffHexa8->computeMass();
            elemNodeIndexes = elem->indexes();
            int numNodes = (int)elemNodeIndexes.size();
            for(auto it = elemNodeIndexes.begin(); it != elemNodeIndexes.end(); it++){
                mass.data(*it, 0) += elemMass/numNodes;
                mass.data(*it, 1) += elemMass/numNodes;
                mass.data(*it, 2) += elemMass/numNodes;
            }
        }//Loop on elements
        mass.writeToFile<double>();
    }
    for(int ct = 0; ct < mesh->numNodes(); ct++) for(int ctKort = 0; ctKort < 3; ctKort++)
        demp.data(ct, ctKort) = ksi*mass.data(ct, ctKort);
}

void getDispl(const double t, double * displ, double ampX, double freqX, double transT)
{
    double omegaX = 4*std::atan(1)*freqX;
    double transAmp = 1;
    if ( t < transT )
        transAmp = t/transT;
    displ[0] = transAmp*(ampX*std::sin(omegaX*t) + ampX/2*std::sin(omegaX*2*t));
    displ[1] = transAmp*(ampX*std::sin(omegaX/3*t) + ampX/5*std::sin(omegaX*3*t));
    displ[2] = 0;
}
