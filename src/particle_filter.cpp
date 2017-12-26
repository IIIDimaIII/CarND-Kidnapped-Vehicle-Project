/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

vector<double> get_map_coords(double x_part, double y_part,double x_obs, double y_obs, double theta);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).	
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);	
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);	
	num_particles = 200;
	
	for (int i = 0; i < num_particles; ++i) {		
		Particle p;			
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1./ num_particles;
		
		particles.push_back(p);		
		weights.push_back(p.weight);
	}
	is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	
	for (int i = 0; i < num_particles; i++) {
		double pred_x, pred_y, pred_theta;		
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		if(fabs(yaw_rate > 0.001)){
			pred_x = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			pred_y = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			pred_theta = theta + yaw_rate * delta_t;
		}
		else{
			pred_x = x + velocity * cos(theta) * delta_t;
			pred_y = y + velocity * sin(theta) * delta_t;
			pred_theta = theta;
		}

		//add noise - increase std_dev to capture motion model uncertainty
		normal_distribution<double> dist_x(pred_x, std_pos[0]*4);	
		normal_distribution<double> dist_y(pred_y, std_pos[1]*4);
		normal_distribution<double> dist_theta(pred_theta, std_pos[2]*4);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	vector<int> associated_pred;
	for(int obs_id =0; obs_id< observations.size();obs_id++){		
		int a1 = 0; //current index of prediction with lowest distance to the observation
		double minimum_distance = 10000;		
		for(int pred_id = 0; pred_id<predicted.size(); pred_id++){
			double x1 = predicted[pred_id].x;
			double y1 = predicted[pred_id].y;
			double x2 = observations[obs_id].x;
			double y2 = observations[obs_id].y;
			double d = dist(x1,y1,x2,y2);			
			if(d < minimum_distance){
				a1 = pred_id;
				minimum_distance = d;
			}
		}		
		associated_pred.push_back(a1);
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++){
		//get particles pred_coords
		double x_part = particles[i].x;
		double y_part = particles[i].y;
		double theta = particles[i].theta;
		double w_p = 1;

		//for convert each observation to map
		vector<LandmarkObs> obs_in_map_coord;
		for(int ob_index = 0; ob_index < observations.size(); ob_index++){			
			double x_obs = observations[ob_index].x;
			double y_obs = observations[ob_index].y;
			vector<double> converted_coords = get_map_coords(x_part, y_part,x_obs, y_obs, theta);

			//find nearest landmark
			double min_dist = sensor_range;
			double min_lx;
			double min_ly;
			for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
				double land_x = map_landmarks.landmark_list[j].x_f;
				double land_y = map_landmarks.landmark_list[j].y_f;
				double d = dist(converted_coords[0],converted_coords[1],land_x, land_y);

				if(d < min_dist){
					min_dist = d;
					min_lx = land_x;
					min_ly = land_y; 
				}
			}
			
			double e_pow = -(pow((converted_coords[0] - min_lx)/std_landmark[0],2) +
								 pow((converted_coords[1] - min_ly)/std_landmark[1],2)) / 2.;
			w_p *= 1. / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(e_pow);
		}
		particles[i].weight = w_p;
		weights[i] = w_p;
		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> discr_distr (weights.begin(),weights.end());

	std::vector<Particle> resampled_particles;
	for(int i = 0; i < num_particles; i++){
		resampled_particles.push_back(particles[discr_distr(gen)]);
	}
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

vector<double> get_map_coords(double x_part, double y_part,double x_obs, double y_obs, double theta)
{ 	vector<double> out; 
	double x_map= x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
	double y_map= y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
	out.push_back(x_map);
	out.push_back(y_map);
	return out;
}
