function [p_class, p_feature_class] = NaiveModel(my_set, my_res, CLASSES, LEVELS)
    % Computes P(c) and P(X | c)
    
    % Inputs: my_set: data set of attributes
    %         my_res: results of data set
    %         CLASSES: number of possible results
    %         LEVELS: levels of each attribute
    % Outputs: p_class: P(c)
    %          p_feature_class: P(X | c)
    
    [EXAMPLES, FEATURES] = size(my_set);
    max_n_levels=max(LEVELS); 
    
    % To count the number of instances of classes
    N_c = zeros(CLASSES, 1); 
    % To count the number of instances of features for classes
    N_jc = zeros(FEATURES, max_n_levels, CLASSES);  
    
    %% Analyze data set
    for e = 1:EXAMPLES % data set rows
        c=my_res(e); 
        N_c(c)=N_c(c)+1;
        for f = 1:FEATURES 
            for l = 1:LEVELS(f) 
                if my_set(e,f)==l
                    N_jc(f,l,c)= N_jc(f,l,c)+1; 
                end
            end
        end
    end
    
    
    %% Compute probabilities
    
    %% P(c)
    p_class = zeros(CLASSES, 1);
    
    % Counts the ratio between the number of examples of each class over 
    % the total number of examples
    for c = 1:CLASSES
        p_class(c) = N_c(c) / EXAMPLES;
    end
    
    %% P(X | c)
    p_feature_class = zeros(FEATURES, max_n_levels, CLASSES);

    % Counts the ratio between the number of appareances of a feature for
    % each class over the total number of examples of that class
    for c = 1:CLASSES 
        for f = 1:FEATURES 
            for l = 1:max_n_levels
                p_feature_class(f, l, c) = (N_jc(f, l, c)) / ( N_c(c));
            end
        end
    end    
    
    % Checks that probabilities sum up to (about) 1.
    for f=1:FEATURES
        for c=1:CLASSES
            if sum(p_feature_class(f,:,c))>=1.001 || sum(p_feature_class(f,:,c))<0.999
                disp('Probability may be wrong');
            end
        end
    end
end
