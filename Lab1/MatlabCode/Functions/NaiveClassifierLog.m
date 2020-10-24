function p_class_feature = NaiveClassifierLog(test_set, p_c, p_feature_class, CLASSES)
    % Computes P(c | X) of a test set given the already computed P(c) and
    % P(X | c) for the training data set.
    
    % Inputs: test_set: data set of attributes
    %         p_c: P(c)
    %         p_feature_class: P(X | c)
    %         N_TEST: number of examples of the test set
    %         CLASSES: number of possible results
    % Outputs: p_class_feature: P(c | X)
    
    %% P(c | X)
    N_TEST=size(test_set,1);
    p_class_feature = zeros(N_TEST, CLASSES); % nx2
    [N_TEST, FEATURES] = size(test_set);
  
    sum=zeros(N_TEST,CLASSES);

    for e = 1:N_TEST % data set rows
      for f=1:FEATURES % data set cols
          l=test_set(e,f); % level taken from data set
          for c=1:CLASSES
            sum(e,c) = sum(e,c) + log(p_feature_class(f,l,c));
          end
      end

      for c = 1:CLASSES
         p_class_feature(e, c) = sum(e,c) + log(p_c(c));
      end
    end        
  
end
