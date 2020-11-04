function W = multiDimLinReg(X,t)
% Computes the parameters vector for the multi dimensional linear regression
% problem
% Inputs: X: data matrix
%         t: target vector
% Outputs: W: weights matrix
    W=pinv(X'*X)*X'*t;
end

