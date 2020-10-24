function [p_class, p_feature_class] = NaiveModelSmooth(my_set, my_res, CLASSES, LEVELS)
    % Computes P(c) and P(X | c) with a smoothing factor
    
    % Inputs: my_set: data set of attributes
    %         my_res: results of data set
    %         CLASSES: number of possible results
    %         LEVELS: levels of each attributes
    % Outputs: p_class: P(c)
    %          p_feature_class: P(X | c)
    
    [EXAMPLES, FEATURES] = size(my_set);
    max_n_levels=max(LEVELS);
    
    % To count the number of yes and no
    N_c = zeros(CLASSES, 1); 
    N_jc = zeros(FEATURES, max_n_levels, CLASSES);
    
    %% Analyze data set
    for e = 1:EXAMPLES 
        c = my_res(e); 
        N_c(c) = N_c(c)+1; 
        for f = 1:FEATURES
            for l = 1:LEVELS(f) 
                if my_set(e,f)==l
                    N_jc(f,l,c) = N_jc(f,l,c)+1; 
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
    a=1;
    for c = 1:CLASSES 
        for f = 1:FEATURES 
            for l = 1:max_n_levels
                p_feature_class(f, l, c) = (N_jc(f, l, c) + a) / ( N_c(c) + a * LEVELS(f));
            end
        end
    end    
end
