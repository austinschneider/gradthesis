#include "gtest/gtest.h"
#include <limits.h>
#include "PROPOSAL/Integral.h"
#include "math.h"
#include <iostream>

double Testfkt(double r){
  return exp(r);
}

double Testexp(double r){
    return exp(r);
}

bool relErr(double Is, double HasToBe, double RelError){
    return fabs((Is - HasToBe)/HasToBe)<RelError;
}

TEST(Comparison , Comparison_equal ) {
    Integral A;
    Integral B;
    EXPECT_TRUE(A==B);
    Integral* C = new Integral(5,20,1e-5);
    Integral* D = new Integral(5,20,1e-5);
    EXPECT_TRUE(*C==*D);
    C->Integrate(0,3,Testfkt,1);
    D->Integrate(0,3,Testfkt,1);
    EXPECT_TRUE(*C==*D);

}

TEST(Comparison , Comparison_not_equal ) {
    Integral A;
    Integral B(5,10,1e-5);
    EXPECT_TRUE(A!=B);
    Integral* C = new Integral(5,20,1e-5);
    Integral* D = new Integral(5,20,1e-5);
    C->Integrate(1,3,Testfkt,1);
    D->Integrate(0,3,Testfkt,1);
    EXPECT_TRUE(*C!=*D);
    Integral* E = new Integral(1,20,1e-5);
    Integral* F = new Integral(5,20,1e-5);
    EXPECT_TRUE(*E!=*F);


}

TEST(Assignment , Copyconstructor ) {
    Integral A;
    Integral B =A;

    EXPECT_TRUE(A==B);

}

TEST(Assignment , Copyconstructor2 ) {
    Integral A;
    Integral B(A);

    EXPECT_TRUE(A==B);

}

TEST(Assignment , Operator ) {
    Integral A;
    A.Integrate(0,3,Testfkt,1);
    Integral B(8,40,1e-9);

    EXPECT_TRUE(A!=B);

    B=A;

    EXPECT_TRUE(A==B);
}

TEST(Assignment , Swap ) {
    Integral A;
    Integral B;
    EXPECT_TRUE(A==B);
    Integral* C = new Integral(5,20,1e-5);
    Integral* D = new Integral(5,20,1e-5);
    EXPECT_TRUE(*C==*D);
    C->Integrate(0,3,Testfkt,1);
    D->Integrate(0,3,Testfkt,1);
    EXPECT_TRUE(*C==*D);

    A.swap(*C);
    EXPECT_TRUE(A==*D);
    EXPECT_TRUE(*C==B);


}

TEST(IntegralValue , Zero_to_Three_of_xx ) {
    Integral* Int = new Integral();
    ASSERT_NEAR(Int->Integrate(0,3,Testfkt,1),exp(3)-1 , (exp(3)-1)*1E-6);
    delete Int;
}

TEST(IntegralValue, EqualBorders) {
    Integral* Int = new Integral();

    EXPECT_EQ(Int->Integrate(3,3,Testfkt,1),0);

    delete Int;
}

TEST(IntegralValue, SmallError) {
    Integral* Int = new Integral();

    ASSERT_NEAR(   Int->Integrate(0,3,Testexp,1),exp(3)-1
                            ,(exp(3)-1)*1.e-6);

    delete Int;
}

TEST(IntegralValue, FloatEqual) {
    Integral* Int = new Integral();
    //Last 4 digits can differ. relError<1E-4
    ASSERT_FLOAT_EQ(Int->Integrate(0,3,Testexp,1),exp(3)-1);

    delete Int;
}

TEST(IntegralValue, MultiplePrecisions) {
    double xmin=0,xmax=3;
    double  ExactIntegral=exp(3)-1;
    double  CalcIntegral=0;

    double precision = 1E-5;
    for(double precision = 1E-5; precision>1E-16;precision/=10){

        Integral* Int = new Integral(5,20,precision);
        CalcIntegral = Int->Integrate(xmin,xmax,Testexp,1);

        ASSERT_NEAR(CalcIntegral,ExactIntegral, ExactIntegral*precision);

        delete Int;
    }
}

TEST(IntegralValue, IntegrateWithSubstitution) {
    double xmin=2,xmax=4;
    double  ExactIntegral=exp(xmax)-exp(xmin);
    double  CalcIntegral=0;

    double precision = 1E-5;
    for(double precision = 1E-5; precision>1E-11;precision/=10){

        Integral* Int = new Integral(5,20,precision);
        CalcIntegral = Int->Integrate(xmin,xmax,Testexp,3,2.);

        ASSERT_NEAR(CalcIntegral,ExactIntegral, ExactIntegral*precision);

        delete Int;
    }
}

TEST(IntegralValue, IntegrateWithLog) {
    double xmin=2,xmax=4;
    double  ExactIntegral=exp(xmax)-exp(xmin);
    double  CalcIntegral=0;

    double precision = 1E-5;
    for(double precision = 1E-5; precision>1E-11;precision/=10){

        Integral* Int = new Integral(5,20,precision);
        CalcIntegral = Int->Integrate(xmin,xmax,Testexp,4);

        ASSERT_NEAR(CalcIntegral,ExactIntegral, ExactIntegral*precision);

        delete Int;
    }
}

TEST(IntegralValue, IntegrateWithLogSubstitution) {
    double xmin=2,xmax=4;
    double  ExactIntegral=exp(xmax)-exp(xmin);
    double  CalcIntegral=0;

    double precision = 1E-5;
    for(double precision = 1E-5; precision>1E-6;precision/=10){

        Integral* Int = new Integral(5,20,precision);
        CalcIntegral = Int->Integrate(xmin,xmax,Testexp,5,2.);

        ASSERT_NEAR(CalcIntegral,ExactIntegral, ExactIntegral*precision);

        delete Int;
    }
}

//TEST(IntegralValue, HasToFail) {
//    EXPECT_TRUE(false);
//}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
