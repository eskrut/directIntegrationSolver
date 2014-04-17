#include "prepareMesh.h"
#include "sbf.h"
#include <cmath>
#include <memory>

#include <thread>

sbfMesh *createMesh(int discretParam)
{
    report("Recreating mesh");
    sbfMesh *mesh = new sbfMesh;

    std::unique_ptr<sbfMesh> c(new sbfMesh);
    std::unique_ptr<sbfMesh> plus(new sbfMesh);

    auto threadC = std::thread([&](){
        sbfMesh tmp;
        //Construct C letter
        std::unique_ptr<sbfMesh> roundInner(sbfMesh::makeCylinderPart(1, 2, 0, 2*std::atan(1), 0, 1, discretParam, 2*discretParam, discretParam));
        std::unique_ptr<sbfMesh> roundOuter(sbfMesh::makeCylinderPart(5, 6, 0, 2*std::atan(1), 0, 1, discretParam, 8*discretParam, discretParam));
        std::unique_ptr<sbfMesh> round(new sbfMesh);
        round->addMesh(roundInner.get());
        round->addMesh(roundOuter.get());
        std::unique_ptr<sbfMesh> part0(sbfMesh::makeBlock(4, 1, 1, 4*discretParam, discretParam, discretParam));
        std::unique_ptr<sbfMesh> part1(sbfMesh::makeBlock(1, 13, 1, discretParam, 13*discretParam, discretParam));
        std::unique_ptr<sbfMesh> part2(sbfMesh::makeBlock(1, 3, 1, discretParam, 3*discretParam, discretParam));
        std::unique_ptr<sbfMesh> part3(sbfMesh::makeBlock(1, 3, 1, discretParam, 3*discretParam, discretParam));
        std::unique_ptr<sbfMesh> part4(sbfMesh::makeBlock(3, 1, 1, 3*discretParam, discretParam, discretParam));
        c->addMesh(part0.get());
        part0->translate(0, 4, 0);
        c->addMesh(part0.get());
        part0->translate(0, 16, 0);
        c->addMesh(part0.get());
        part0->translate(0, 4, 0);
        c->addMesh(part0.get());
        tmp = *round.get();
        tmp.translate(4, 19, 0);
        c->addMesh(tmp);
        tmp = *round.get();
        tmp.rotate(0, 0, 2*std::atan(1));
        tmp.translate(0, 19, 0);
        c->addMesh(tmp);
        tmp = *round.get();
        tmp.rotate(0, 0, 4*std::atan(1));
        tmp.translate(0, 6, 0);
        c->addMesh(tmp);
        tmp = *round.get();
        tmp.rotate(0, 0, 6*std::atan(1));
        tmp.translate(4, 6, 0);
        c->addMesh(tmp);
        part1->translate(-2, 6, 0);
        c->addMesh(part1.get());
        part1->translate(-4, 0, 0);
        c->addMesh(part1.get());
        tmp = *part2.get();
        part2->translate(4, 0, 0);
        tmp.addMesh(part2.get());
        part2->translate(-4, 0, 0);
        part2->rotate(0, 0, -2*std::atan(1));
        part2->translate(1, 3, 0);
        tmp.addMesh(part2.get());
        tmp.translate(5, 6, 0);
        c->addMesh(tmp);
        tmp.translate(-5, -6, 0);
        tmp.rotate(0, 0, 4*std::atan(1));
        tmp.translate(10, 19, 0);
        c->addMesh(tmp);
        part3->translate(0, 1, 0);
        c->addMesh(part3.get());
        part3->translate(3, 0, 0);
        c->addMesh(part3.get());
        part3->translate(0, 20, 0);
        c->addMesh(part3.get());
        part3->translate(-3, 0, 0);
        c->addMesh(part3.get());
        part4->translate(-5, 6, 0);
        c->addMesh(part4.get());
        part4->translate(0, 4, 0);
        c->addMesh(part4.get());
        part4->translate(0, 4, 0);
        c->addMesh(part4.get());
        part4->translate(0, 4, 0);
        c->addMesh(part4.get());
    });

    auto threadPlus = std::thread([&](){
        sbfMesh tmp;
        //Construct plus sign
        std::unique_ptr<sbfMesh> part10(sbfMesh::makeBlock(11, 1, 1, 11*discretParam, discretParam, discretParam));
        std::unique_ptr<sbfMesh> part11(sbfMesh::makeBlock(1, 3, 1, discretParam, 3*discretParam, discretParam));
        std::unique_ptr<sbfMesh> part12(sbfMesh::makeBlock(10, 1, 1, 10*discretParam, discretParam, discretParam));
        part11->translate(0, 1, 0);
        part12->translate(0, 4, 0);
        tmp = *part10.get();
        tmp.addMesh(part11.get());
        part11->translate(3, 0, 0);
        tmp.addMesh(part11.get());
        part11->translate(3, 0, 0);
        tmp.addMesh(part11.get());
        part11->translate(3, 0, 0);
        tmp.addMesh(part11.get());
        tmp.addMesh(part12.get());
        tmp.translate(-12.5, -2.5, 0);
        plus->addMesh(tmp);
        tmp.rotate(0, 0, 2*std::atan(1));
        plus->addMesh(tmp);
        tmp.rotate(0, 0, 2*std::atan(1));
        plus->addMesh(tmp);
        tmp.rotate(0, 0, 2*std::atan(1));
        plus->addMesh(tmp);
        plus->translate(2.5, 12.5, 0);
    });

    threadC.join();
    threadPlus.join();

    c->applyToAllElements([](sbfElement &elem){elem.setMtr(2);});
    mesh->addMesh(c.get());
    plus->applyToAllElements([](sbfElement &elem){elem.setMtr(3);});
    plus->translate(23, 0, 0);
    mesh->addMesh(plus.get(), false, false);
    plus->applyToAllElements([](sbfElement &elem){elem.setMtr(4);});
    plus->translate(28, 0, 0);
    mesh->addMesh(plus.get(), false, false);

    mesh->translate(0, 1, 0);
    std::unique_ptr<sbfMesh> footage(sbfMesh::makeBlock(56, 1, 1, 56*discretParam, discretParam, discretParam));
    footage->applyToAllElements([](sbfElement &elem){elem.setMtr(1);});
    mesh->addMesh(footage.get());

    return mesh;
}

sbfStiffMatrixBlock3x3 *createStiffMatrix(sbfMesh *mesh, bool recompute)
{
    sbfPropertiesSet *propSet = new sbfPropertiesSet;
    propSet->addMaterial(sbfMaterialProperties::makeMPropertiesGold());
    propSet->addMaterial(sbfMaterialProperties::makeMPropertiesSilver());
    propSet->addMaterial(sbfMaterialProperties::makeMPropertiesGold());
    propSet->addMaterial(sbfMaterialProperties::makeMPropertiesBronze());

    sbfStiffMatrixBlock3x3 *stiff = new sbfStiffMatrixBlock3x3;
    stiff->setMesh(mesh);
    stiff->setPropSet(propSet);
    stiff->updateIndexesFromMesh();

    if( recompute || !stiff->read("stif.sba")) {
        report("Recomputing stiffness matrix");
        stiff->compute();
        stiff->write("stif.sba");
    }

    return stiff;
}

