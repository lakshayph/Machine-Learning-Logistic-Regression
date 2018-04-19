function logistic_regression(training_file, d, test_file)

%LINEAR Summary of this function goes here
%   Detailed explanation goes here

    training_data  = importdata(training_file);
    test_data = importdata(test_file);
    degree = str2double(d);
    num_columns = size(training_data,2); %number of columns of training_data
    if(degree == 1)
        training = zeros(size(training_data,1),num_columns);
    else
        training = zeros(size(training_data,1),(2*num_columns)-1);
    end
    
    if(degree == 1)
        test = zeros(size(test_data,1),num_columns);
    else
        test = zeros(size(test_data,1),(2*num_columns)-1);
    end
    classes = zeros(size(training_data,1),1);
    classes_test = zeros(size(test_data,1),1);
    
    %initializing classes to 0 and 1
    for k=1:size(training_data,1)
        if(training_data(k,num_columns) == 1)
            classes(k,1)=1;
        else
            classes(k,1)=0;
        end
    end
    
    for z=1:size(test_data,1)
        if(test_data(z,num_columns) == 1)
            classes_test(z,1)=1;
        else
            classes_test(z,1)=0;
        end
    end
    
    %matrix formation on the basis of degree i.e. phi matrix for training data
    if (degree ==1)
        training(:,1) = 1;
        for j=2:num_columns
            training(:,j) = power(training_data(:,j-1),1);
        end
    else 
        training(:,1) = 1;
        for j=2:2:2*(num_columns-1)
            training(:,j) = power(training_data(:,j/2),1);
            training(:,j+1) = power(training_data(:,j/2),2);
        end
    end
    
        %matrix formation on the basis of degree i.e. phi matrix for test data
    if (degree ==1)
        test(:,1) = 1;
        for j=2:num_columns
            test(:,j) = power(test_data(:,j-1),1);
        end
    else 
        test(:,1) = 1;
        for j=2:2:2*(num_columns-1)
            test(:,j) = power(test_data(:,j/2),1);
            test(:,j+1) = power(test_data(:,j/2),2);
        end
    end
    
    training_transpose = training'; %transpose of phi matrix for training data
    
    weight = zeros(size(training,2),1); %initializing weight matrix to zero
    
    weighted_sum = 1;
    weighted_entropy = 1;
    
    while (weighted_sum > 0.001 && weighted_entropy > 0.001)
        output = logsig(training*weight); %output value for training data using sigmoid function
    
        diagonal_matrix = diag((output.*(1-output)),0); %creating R matrix
    
        element1 = training_transpose*diagonal_matrix;
        element2 = inv(element1*training);
        element3 = element2*training_transpose;
        element4 = element3*(output - classes);
        
        entro1_old = classes.*(log(output));
        entro2_old = (1-classes).*(log(1-output));
        entropy_old = entro1_old + entro2_old;
        
        weights_old = weight;
        weight = weight - element4; %calculation of new weights
        weight_difference = abs(weights_old - weight);
        weighted_sum = sum(weight_difference);
        
        output_new = logsig(training*weight);
        entro1_new = classes.*(log(output_new));
        entro2_new = (1-classes).*(log(1-output_new));
        entropy_new = entro1_new + entro2_new; %calculation of cross entropy error
        
        entropy_difference = abs(entropy_old - entropy_new);
        weighted_entropy = sum(entropy_difference);
    end
    
    %printing the value of weights
    if (degree==1)
        for x=1:size(weight,1)
            fprintf('w%d=%.4f\n',x-1,weight(x));
        end
    else 
        for x=1:size(weight,1)
            fprintf('w%d=%.4f\n',x-1,weight(x));
        end
    end
    
    %code for test data
    test_output = logsig(test*weight);
    
    predicted = zeros(size(test_output,1),1);
    accuracy = zeros(size(test_output,1),1);
   
    for q = 1:size(test_output,1)
        if(test_output(q,1) < 0.5)
            predicted(q,1) = 0;
        else
            predicted(q,1) = 1;
        end
    end
    
    for q = 1:size(test_output,1)
        if(test_output(q,1) == 0.5)
            accuracy(q,1) = 0.5;
        elseif (predicted(q,1) == classes_test(q,1))
            accuracy(q,1) = 1;
        elseif (predicted(q,1) == classes_test(q,1))
            accuracy(q,1) = 0;
        end
    end
    
    for w = 1:size(test_output,1)
        if(test_output(w,1)>0.5)
            fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',w-1, predicted(w,1), test_output(w,1), classes_test(w,1), accuracy(w,1));
        else
            fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',w-1, predicted(w,1), 1 - test_output(w,1), classes_test(w,1), accuracy(w,1));    
        end
    end 
    
    classification_accuracy = mean(accuracy);
    fprintf('classification accuracy=%6.4f\n', classification_accuracy);
end