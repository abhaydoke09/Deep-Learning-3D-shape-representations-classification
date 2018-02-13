function [ loss ] = cost( y, yp, model, weight_decay)
%COST Summary of this function goes here
%   Detailed explanation goes here
   
    y_1 = model.outputs{model.num_layers}.*(y==1);
    y_0 = model.outputs{model.num_layers}.*(y==0);
    %disp(model.outputs{model.num_layers});
    y_1 = y_1(y_1~=0);
    y_0 = y_0(y_0~=0);
        
    cross_entropy_loss = -1.0*(sum(log(y_1)) + sum(log(1-y_0)));
    
    regularization_loss = 0.0; 
    for layer_id=2:model.num_layers
        regularization_loss = regularization_loss + sum(sum(model.param{layer_id}.^2));
    end
    
    regularization_loss = weight_decay*regularization_loss;
    
    loss = cross_entropy_loss + regularization_loss;
    %disp(loss);
end

