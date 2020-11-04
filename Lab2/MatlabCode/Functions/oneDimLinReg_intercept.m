function [w1,w0] = oneDimLinReg_intercept(x,t)
% Computes the parameters vector for the one dimensional linear regression
% with intercept problem
% Inputs: x: data
%         t: target
% Outputs: w1: slope
%          w0: intercept
    x_mean=mean(x);
    t_mean=mean(t);

    x=x-x_mean;
    t=t-t_mean;
    w1=sum(x.*t) / sum(x.^2);
    w0=t_mean-w1*x_mean;
end

