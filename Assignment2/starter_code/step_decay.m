function [ learning_rate ] = step_decay( initial_learning_rate, decay, epoch )
% This function is used for calculating the learning rate based on
% step_decay method. 

learning_rate = initial_learning_rate * 1/(1 + decay*epoch);
end

