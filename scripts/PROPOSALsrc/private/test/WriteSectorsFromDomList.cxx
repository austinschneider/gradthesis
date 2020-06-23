#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;


void AddSector(ofstream& out,
               string name,
               string geometry,
               double x0,
               double y0,
               double z0,
               double radius,
               double height,
               int hirarchy,
               string medium,
               double density)
{
    out<<name<<endl;
    out<<"sector"<<endl;

    if(geometry.compare("sphere")==0)
    {
        out<<"\t"<<geometry<<" "<<x0<<" "<<y0<<" "<<z0<<" "<<radius<<" 0"<<endl;
    }
    if(geometry.compare("cylinder")==0)
    {
        out<<"\t"<<geometry<<" "<<x0<<" "<<y0<<" "<<z0<<" "<<radius<<" 0 "<<height<<endl;
    }
    out<<"\t"<<"hirarchy "<<hirarchy<<endl;
    out<<"\t"<<"medium "<<medium<<" "<<density<<endl;

}


int main()
{
    string dom_list_file_name = "resources/Icecube_geometry.20110414.complete.txt";
    ofstream out;
    out.open("resources/sectors_from_dom_list");

    int string  = 0;
    int dom     = 0;
    double x    = 0;
    double y    = 0;
    double z    = 0;
    double string_z0    =   0;

    double height_of_last_dom_on_string;

    ifstream in;
    in.open(dom_list_file_name.c_str());

    stringstream ss;

    while(in.good())
    {
        in>>string>>dom>>x>>y>>z;

        if(dom==60)
        {
            height_of_last_dom_on_string    =   z;
        }

        if(dom>61)
            continue;

        ss<<"# sector for dom "<<dom<<" on string "<<string;

        if(dom != 61)
            AddSector(out,ss.str(),"sphere",x,y,z,0.25,0,4,"uranium",1);

        if(dom==61)
        {
            ss.str("");
            ss.clear();

            ss<<"# sector for string "<<string;

            string_z0 = height_of_last_dom_on_string + (z - height_of_last_dom_on_string)/2;
            AddSector(out,ss.str(),"cylinder",x,y,string_z0,0.05,(z - height_of_last_dom_on_string),3,"copper",1);

            ss.str("");
            ss.clear();

            ss<<"# sector for hole ice of string "<<string;

            AddSector(out,ss.str(),"cylinder",x,y,string_z0,0.05,(z - height_of_last_dom_on_string),2,"ice",0.6);


        }

        ss.str("");
        ss.clear();

    }

}
