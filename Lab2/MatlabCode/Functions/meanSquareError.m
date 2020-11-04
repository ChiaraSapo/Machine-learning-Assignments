function objective = meanSquareError(x,t,w1,w0,problem)
% Computes the mean square error of the linear regression
% Inputs: x: data
%         t: target
%         w1: slope
%         w0: intercept
%         problem: type of linear regression problem. can be 1 for 1-D, 2 for multi-D
% Outputs: objective: objective function
    N = length(x);
    
    % 1-D problem
    if problem == 1
        y = w1*x + w0; 
        objective = (1/N)*sum((t-y).^2); 
    end
    % multi-D problem
    if problem == 2 
        y = x*w1; 
        objective = immse(t,y); 
    end

end

