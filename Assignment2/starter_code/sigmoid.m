function [ x ] = sigmoid( x )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

x = 1./(1 + exp(-1*(x)));
end

