/*! \file   EnergyCutSettings.h
*   \brief  Headerfile for the energy cuts routines.
*
*   For more details see the class documentation.
*
*   \date   2013.03.15
*   \author Jan-Hednrik Köhne
*/

#ifndef ENERGYCUTSETTINGS_H_
#define ENERGYCUTSETTINGS_H_

#include <iostream>


/**
 * \brief Class containing the energy cut such as ecut and vcut.
 *
 * The cuts ecut and vcut to be used must satisfy \f$e_{cut}\geq 0\f$ and
 * \f$0 \leq v_{cut} \leq 1\f$; If both values satisfy these
 * inequalities, the lower from \f$ E \cdot v_{cut} \f$ and \f$ e_{cut}\f$
 * will be used. If only one value satisfies the inequalities,
 * only that one value is used. If both values are outside these intervals,
 * \f$v_{cut}=1\f$ is assumed.
 */


class EnergyCutSettings
{

private:

    double ecut_;
    double vcut_;

public:

 //----------------------------------------------------------------------------//
    // Constructors

    EnergyCutSettings();
    EnergyCutSettings(const EnergyCutSettings&);
    EnergyCutSettings& operator=(const EnergyCutSettings& energyCutSettings);
    EnergyCutSettings(double ecut, double vcut);
    bool operator==(const EnergyCutSettings &energyCutSettings) const;
    bool operator!=(const EnergyCutSettings &energyCutSettings) const;
    friend std::ostream& operator<<(std::ostream& os, EnergyCutSettings const& cut_settings);

//----------------------------------------------------------------------------//

    //Memberfunction

    /*!
     * This function returns the lower from
     * \f$ E \cdot v_{cut} \f$ and \f$ e_{cut}\f$
     *
     *
     * \param   energy      energy of the particle
     * \return  cut
     */
    double GetCut(double energy);

//----------------------------------------------------------------------------//

    void swap(EnergyCutSettings &energyCutSettings);

 //----------------------------------------------------------------------------//
    //Getter

    double GetEcut() const
    {
        return ecut_;
    }
//----------------------------------------------------------------------------//
    double GetVcut() const
    {
        return vcut_;
    }
//----------------------------------------------------------------------------//

    //Setter
    void SetEcut(double ecut);
    void SetVcut(double vcut);

};

#endif /* ENERGYCUTSETTINGS_H_ */
