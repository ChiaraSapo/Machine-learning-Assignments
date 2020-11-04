function w = oneDimLinReg(x,t)
% Computes the parameters vector for the one dimensional linear regression
% problem
% Inputs: x: data
%         t: target
% Outputs: w: weights
    w=sum(x.*t) / sum(x.^2);
    
end

