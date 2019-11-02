#ifndef _VESTIBULAR_SENSOR_CPP
#define _VESTIBULAR_SENSOR_CPP

#include "iostream"
#include "vestibularSensor.h"
#include "neuron.h"

VESTIBULAR_SENSOR::VESTIBULAR_SENSOR(int myID, int evalPeriod) {

	ID = myID;

	w = new double[evalPeriod];
	x = new double[evalPeriod];
	y = new double[evalPeriod];
	z = new double[evalPeriod];


	for (int i = 0; i < 4; i++) {
	    mySensorNeurons[i] = NULL;
	}
}

VESTIBULAR_SENSOR::~VESTIBULAR_SENSOR(void) {

}

void VESTIBULAR_SENSOR::Connect_To_Sensor_Neuron(NEURON *sensorNeuron) {

    mySensorNeurons[ sensorNeuron->Get_Sensor_Value_Index() ] = sensorNeuron;
}

int  VESTIBULAR_SENSOR::Get_ID(void) {

        return ID;
}

void VESTIBULAR_SENSOR::Poll(dBodyID body, int t) {

        const dReal *q = dBodyGetQuaternion(body);

        w[t] = q[0];
        x[t] = q[1];
        y[t] = q[2];
        z[t] = q[3];
}

void VESTIBULAR_SENSOR::Update_Sensor_Neurons(int t) {

    if ( mySensorNeurons[0] )

        mySensorNeurons[0]->Set( w[t] );

    else if ( mySensorNeurons[1] )

        mySensorNeurons[1]->Set( x[t] );

    else if ( mySensorNeurons[2] )

        mySensorNeurons[2]->Set( y[t] );

    else if ( mySensorNeurons[3] )

        mySensorNeurons[3]->Set( z[t] );
}

void VESTIBULAR_SENSOR::Write_To_Python(int evalPeriod) {

        char outString[1000000];

        sprintf(outString,"%d %d ",ID,4);

        for ( int  t = 0 ; t < evalPeriod ; t++ ) {
            sprintf(outString,"%s %f %f %f %f ",outString,w[t], x[t], y[t], z[t]);
        }

        sprintf(outString,"%s \n",outString);

        std::cout << outString;
}

static void toEulerAngle(const dReal *q, double& roll, double& pitch, double& yaw)
{
	// roll (x-axis rotation)
	double sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3]);
	double cosr_cosp = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
	roll = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q[0] * q[2] - q[3] * q[1]);
	if (fabs(sinp) >= 1)
		pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		pitch = asin(sinp);

	// yaw (z-axis rotation)
	double siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2]);
	double cosy_cosp = +1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
	yaw = atan2(siny_cosp, cosy_cosp);
}

#endif
