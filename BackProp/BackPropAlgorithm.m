%% Author : Krishna Satish D S

function BackPropAlgorithm()
    %% Dataset parameters
    sepal_l_Min = 4.3;
    sepal_l_Max = 7.9;
    sepal_w_Min = 2.0;
    sepal_w_Max = 4.4;
    petal_l_Min = 1.0;
    petal_l_Max = 6.9;
    petal_w_Min = 0.1;
    petal_w_Max = 2.5;
    %% Desired Outputs
    setosa = [0.1 0.1 0.9]';
    versicolor = [0.1 0.9 0.1]';
    virginica = [0.9 0.1 0.1]';
    %% Read the dataset
    data = load('IrisDataSet.txt');
    sepal_l = data(:,1);
    sepal_w = data(:,2);
    petal_l = data(:,4);
    petal_w = data(:,4);
    %% Normalizing the data
    n_sepal_l = scaledata(sepal_l,sepal_l_Min,sepal_l_Max);
    n_sepal_w = scaledata(sepal_w,sepal_w_Min,sepal_w_Max);
    n_petal_l = scaledata(petal_l,petal_l_Min,petal_l_Max);
    n_petal_w = scaledata(petal_w,petal_w_Min,petal_w_Max);
    n_data = [n_sepal_l, n_sepal_w,  n_petal_l, n_petal_w];
    %% Network parameters
    Number_Input = 4;
    Number_Of_Hidden_Nodes = 5;
    Number_Of_Outputs = 3;
    Learning_Rate = 0.09;
    Iteration=5000;
    Weight_Input_Hidden = rand(Number_Input,Number_Of_Hidden_Nodes);
    Weight_Bias_Hidden = rand(Number_Of_Hidden_Nodes,1);
    Weight_Bias_Output = rand(Number_Of_Outputs,1);
    Weight_Hidden = rand(Number_Of_Hidden_Nodes,Number_Of_Outputs);
    error=zeros(Number_Of_Outputs,Iteration);
    %% Shuffling Training Data 
    Shuffled_Data = [];
    for iterator=1:40
         buffer = [n_data(iterator,:);n_data(50+iterator,:);n_data(100+iterator,:)]';
         Shuffled_Data = [Shuffled_Data buffer];
    end
    %% OutPut Value Prep
    Desired_out_temp = [setosa versicolor virginica];
    Desired_out=repmat(Desired_out_temp,[1 40]);
    %% Training 
    h = waitbar(0,'Training...');
    for iterator_i = 1:Iteration
        for iterator_j = 1:size(Shuffled_Data,2)
            Input_Layer_Weight = Shuffled_Data(:,iterator_j)'*Weight_Input_Hidden;  %% Wx --> Input layer to first hidden layer
            Input_Layer_Weight_Bias = Input_Layer_Weight + Weight_Bias_Hidden';  %% Wx + b
            Hidden_Layer_Input = sigmoid(Input_Layer_Weight_Bias);   %% Sigmoid Activation Function
            Hidden_Layer_temp = (Hidden_Layer_Input *  Weight_Hidden) + Weight_Bias_Output';  %% Hidden Layer Inputs multiplied with Hidden Layer Weights
            Final_Output = sigmoid(Hidden_Layer_temp);
            %% Finding Error
            Err = Desired_out(:,iterator_j)'-Final_Output;   %% Difference b/w output and labelled output
            Delta = (Final_Output.*(1-Final_Output)).* Err;   %% Finding out the Partial derivative 
            %% Updading Weights
            Weight_Hidden=Weight_Hidden+Learning_Rate*Hidden_Layer_Input'*Delta;  %% Updating Hidden Layer Weights
            Weight_Bias_Output = Weight_Bias_Output + (2*Delta');                 %% Updading Biases
            %% Updating Input Layer Weights
            Delta_Hidden = Hidden_Layer_Input'.*(1-Hidden_Layer_Input)'.*(Weight_Hidden*Delta');
            Weight_Input_Hidden=Weight_Input_Hidden+Learning_Rate*(Shuffled_Data(:,iterator_j)*Delta_Hidden');
            Weight_Bias_Hidden = Weight_Bias_Hidden + 2*Delta_Hidden;
        end
            error(:,iterator_i)=Err;
            waitbar(iterator_i / Iteration)
    end
    close(h)
    %% Error Plots
    sse=sum((error(:,1:iterator_i).^2),1);
    plot(sse);
    title('Error Plot-Training');
    xlabel('Number Of Iterations');
    ylabel('Error^2');
    %% Saving Weights
    save('Weights','Weight_Bias_Hidden','Weight_Hidden','Weight_Input_Hidden')
end




