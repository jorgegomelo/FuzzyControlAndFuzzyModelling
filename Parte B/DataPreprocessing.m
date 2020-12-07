% Zeros and poles
num=[2 4];
den=[1 3 4 2];
[numd,dend]=c2dm(num,den,1,'zoh')

% Model Importation
sim('modelosimulinkparteB');

% Empty matrix
matrix = zeros(length(input_random)-3,7); %ignoring 3 inputs

% Matrix filling
matrix(1:end,1) = output_discreet(3:end-1);  %k+2
matrix(1:end,2) = output_discreet(2:end-2);  %k+1
matrix(1:end,3) = output_discreet(1:end-3);  %k
matrix(1:end,4) = input_random(3:end-1);     %k+2
matrix(1:end,5) = input_random(2:end-2);     %k+1
matrix(1:end,6) = input_random(1:end-3);     %k
matrix(1:end,7) = output_discreet(4:end);    %k+3

% Saving matrix
save matrix.dat matrix -ascii

% Spliting data into train and test
train_size = round(0.7*length(input_random));
test_size = round(0.3*length(input_random));
train = matrix(1:train_size,:);
test  = matrix(train_size:end,:);

save train.dat train -ascii

% Subtractive Clustering
% Generates a Sugeno-type FIS using subtractive clustering
subtractiveC = genfis2(train(:,1:6), train(:,7), 0.5);
% FCM clustering
% Generates a FIS using FCM clustering
fcmC = genfis3(train(:,1:6),train(:,7),'sugeno'); 
% Adaptive Neuro-Fuzzy training of Sugeno-type FIS
% 1 means hybrid
subtractiveC_hybrid = anfis(train,subtractiveC,[],[],[],1);
% 0 meansbackpropagation
subtractiveC_backp = anfis(train,subtractiveC,[],[],[],0); 

% Adaptive Neuro-Fuzzy training of Sugeno-type FIS
fcmC_hybrid = anfis(train,fcmC,[],[],[],1);
fcmC_backp  = anfis(train,fcmC,[],[],[],0);

% Saving the fuzzy systems
writefis(subtractiveC_hybrid,'subtractiveC_hybrid');
writefis(subtractiveC_backp,'subtractiveC_backp');
writefis(fcmC_hybrid,'fcmC_hybrid');
writefis(fcmC_backp,'fcmC_backp');

% Perform fuzzy inference calculations
subtractiveC_hybrid_out = evalfis(test(:,1:6),subtractiveC_hybrid);
subtractiveC_backp_out  = evalfis(test(:,1:6),subtractiveC_backp);
fcmC_hybrid_out = evalfis(test(:,1:6),fcmC_hybrid);
fcmC_backp_out  = evalfis(test(:,1:6),fcmC_backp);

% Mean-Squared Error
err_sub_hybrid = immse(test(:,7),subtractiveC_hybrid_out)
err_sub_backp  = immse(test(:,7),subtractiveC_backp_out)
err_fcm_hybrid = immse(test(:,7),fcmC_hybrid_out)
err_fcm_backp  = immse(test(:,7),fcmC_backp_out)


