%% Author : Krishna Satish D S

%% Loading The Weights
load('Weights.mat')    %% Loading Weights
%% Testing 
Test_Input1 = [4.8,3.0,1.4,0.1];          %% Input Corresponds to Setosa
Test_Input2 = [7.3,2.9,6.3,1.8];          %% Input Corresponding to Virginica
Test_Input3 = [6.0,2.9,4.5,1.5];          %% Input Corresponding to versicolor
Test_Hidden_In = Test_Input2*Weight_Input_Hidden;
disp('Runnig Algorithm with ...')
disp(Test_Input1)
Test_Out = sigmoid(Test_Hidden_In+Weight_Bias_Hidden');
Out_Sigmoid_Input = Test_Out * Weight_Hidden;
Out = sigmoid(Out_Sigmoid_Input);
%% Plotting
X = categorical({'Virginica','Versicolor','Setosa'});
X = reordercats(X,{'Virginica','Versicolor','Setosa'});
Y = Out;
bar(X,Y,'FaceColor',[0 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5)