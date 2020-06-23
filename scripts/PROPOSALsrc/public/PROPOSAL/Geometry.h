/**
 * This class contains geometry information
 * for each ProcessCollection
 *
 * @author Jan-Hendrik Köhne
 */

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "PROPOSAL/Particle.h"
#include <string>
#include <utility>
#include <iostream>

class Geometry
{
private:

    double x0_;             //!< x-coordinate of origin ( center of box, cylinder, sphere)
    double y0_;             //!< y-coordinate of origin ( center of box, cylinder, sphere)
    double z0_;             //!< z-coordinate of origin ( center of box, cylinder, sphere)

    double inner_radius_;   //!< for spherical shells or hollow cylinder (0 for sphere / cylinder)
    double radius_;         //!< the radius of the sphere/ cylinder

    double x_;              //!< width of box in x-direction
    double y_;              //!< width of box in y-direction
    double z_;              //!< height of box/cylinder

    std::string object_;    //!< "box" , "cylinder" , "sphere" (sphere and cylinder might be hollow)

    unsigned int hirarchy_; //!< adds a hirarchy of geometry objects to allow crossing geometries

//----------------------------------------------------------------------------//
    /*!
     * This function calculates the distance of the particle position
     * to the border of the sphere (hollow sphere)
     * in direction of the particle trajectory.
     * If the particle trajectory does not have an intersection with the sphere
     * (-1 /-1) is returned
     * If the particle trajectory has two intersections (dist_1 /dist_2) is returned
     * If the particle has one intersection (dist_1 /-1) is returned
     * (one intersection means one intersection in direction of the particle trajectory
     * and one in the opposite direction. Cause we are not intersted in this one. it is set to -1)
     * Note: If the particle is on the spheres border this is not treated as an intersection
     * A particle on the spheres border which moves inside has one intersection,
     * a particle on the spheres border which moves outside has no intersection.
     * Distances smaller then GEOMETRY_PRECISION (1e-9) are also set to -1
     */
    std::pair<double,double> DistanceToBorderSphere(Particle* particle);

//----------------------------------------------------------------------------//
    /*!
     * This function calculates the distance of the particle position
     * to the border of the box in direction of the particle trajectory.
     * If the particle trajectory does not have an intersection with the box
     * (-1 /-1) is returned
     * If the particle trajectory has two intersections (dist_1 /dist_2) is returned
     * If the particle has one intersection (dist_1 /-1) is returned
     * (one intersection means one intersection in direction of the particle trajectory
     * and one in the opposite direction. Cause we are not intersted in this one. it is set to -1)
     * Note: If the particle is on the box' border this is not treated as an intersection
     * A particle on the box' border which moves inside has one intersection,
     * a particle on the box' border which moves outside has no intersection.
     * Distances smaller then GEOMETRY_PRECISION (1e-9) are also set to -1
     */
    std::pair<double,double> DistanceToBorderBox(Particle* particle);

//----------------------------------------------------------------------------//
    /*!
     * This function calculates the distance of the particle position
     * to the border of the cylinder in direction of the particle trajectory.
     * If the particle trajectory does not have an intersection with the cylinder
     * (-1 /-1) is returned
     * If the particle trajectory has two intersections (dist_1 /dist_2) is returned
     * If the particle has one intersection (dist_1 /-1) is returned
     * (one intersection means one intersection in direction of the particle trajectory
     * and one in the opposite direction. Cause we are not intersted in this one. it is set to -1)
     * Note: If the particle is on the cylinders border this is not treated as an intersection
     * A particle on the cylinders border which moves inside has one intersection,
     * a particle on the cylinders border which moves outside has no intersection.
     * Distances smaller then GEOMETRY_PRECISION (1e-9) are also set to -1
     */
    std::pair<double,double> DistanceToBorderCylinder(Particle* particle);

public:

    //Constructors

    Geometry();
    Geometry(const Geometry&);
    Geometry& operator=(const Geometry&);
    bool operator==(const Geometry &geometry) const;
    bool operator!=(const Geometry &geometry) const;
    friend std::ostream& operator<<(std::ostream& os, Geometry const& geometry);

//----------------------------------------------------------------------------//
    //Memberfunctions

    bool IsParticleInside(Particle* particle);

//----------------------------------------------------------------------------//
    bool IsParticleInfront(Particle* particle);

//----------------------------------------------------------------------------//
    bool IsParticleBehind(Particle* particle);

//----------------------------------------------------------------------------//
    std::pair<double,double> DistanceToBorder(Particle* particle);

//----------------------------------------------------------------------------//
    /*!
     * Calculates the distance to the closest approch to the geometry center
     */
    double DistanceToClosestApproach(Particle* particle);

//----------------------------------------------------------------------------//

    void InitBox(double x0, double y0, double z0, double x, double y, double z);

//----------------------------------------------------------------------------//

    void InitSphere(double x0, double y0, double z0, double radius, double inner_radius);

//----------------------------------------------------------------------------//

    void InitCylinder(double x0, double y0, double z0, double radius, double inner_radius, double z);

//----------------------------------------------------------------------------//

    void swap(Geometry &geometry);

//----------------------------------------------------------------------------//
    //Getter
    double GetX0() const {
        return x0_;
    }

    double GetY0() const {
        return y0_;
    }

    double GetZ0() const {
        return z0_;
    }

    double GetX() const {
        return x_;
    }

    double GetY() const {
        return y_;
    }

    double GetZ() const {
        return z_;
    }

    double GetInnerRadius() const {
        return inner_radius_;
    }

    double GetRadius() const {
        return radius_;
    }

    std::string GetObject() const {
        return object_;
    }

    unsigned int GetHirarchy() const {
        return hirarchy_;
    }

//----------------------------------------------------------------------------//
    //Setter
    void SetX0(double x0);
    void SetY0(double y0);
    void SetZ0(double z0);
    void SetX(double x);
    void SetY(double y);
    void SetZ(double z);
    void SetInnerRadius(double inner_radius);
    void SetRadius(double radius);
    void SetObject(std::string object);
    void SetHirarchy(unsigned int hirarchy);
//----------------------------------------------------------------------------//
    //Destructor
    ~Geometry();
};

#endif // GEOMETRY_H
