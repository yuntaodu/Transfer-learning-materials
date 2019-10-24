clear all;

% Set algorithm parameters
options.p = 10;             % keep default
options.sigma = 0.1;        % keep default
options.lambda = 10.0;      % keep default
options.gamma = 1.0;        % [0.1,10]
options.ker = 'rbf';        % 'rbf' | 'linear'
options.Mu = 0.99;           % (0,1)

runs = 1;
result = [];
Accs = [];
plotMatrix = [];
% text dataset experiment
%
textDataSet = ["comp_vs_talk"];
%"comp_vs_rec" , "comp_vs_sci", "comp_vs_talk", "rec_vs_sci", "rec_vs_talk", "sci_vs_talk"
textDataSplitNum = 36;

for i1 = 1:length(textDataSet)
    for index = 1:textDataSplitNum
        data = strcat(textDataSet, '_', num2str(index));
        options.data = data;
        load(strcat('data/',data));
        for j = 0:0.05:1
            options.Mu = j;
            [Acc,Cls,Alpha] = ARRLSB(X_src,X_tar,Y_src,Y_tar,options);
            %           [Acc,Cls,Alpha] = ARRLS(X_src,X_tar,Y_src,Y_tar,options);
%             Accs = [Accs;Acc(end)];
            
%             average = mean(Accs);
%             stderr = std(Accs);
%             result = [result;[average,stderr]];
            plotMatrix = [plotMatrix; [j, Acc(end)]];
            
            fprintf("Progress: %f \n", (((i1-1)*textDataSplitNum + index-1 + j) * 20)/(textDataSplitNum * length(textDataSet) * 20));
        end
        
        plot(plotMatrix(:,1),plotMatrix(:,2));
        xlabel('Mu');
        ylabel('Accuracy');
        hold on;
        plotMatrix = [];
    end
    
    
end

% image dataset experiment

% for dataStr = {'USPS_vs_MNIST'}
%     for iDataset = 1
%         data = strcat(char(dataStr),'_',num2str(iDataset));
%         options.data = data;
%
%         load(strcat('data/',data));
%         for j = 0:0.05:1
%             options.Mu = j;
%             [Acc,Cls,Alpha] = ARRLSB(X_src,X_tar,Y_src,Y_tar,options);
% %           [Acc,Cls,Alpha] = ARRLS(X_src,X_tar,Y_src,Y_tar,options);
%             Accs = [Accs;Acc(end)];
%
%             average = mean(Accs);
%             stderr = std(Accs);
%             result = [result;[average,stderr]];
%             plotMatrix = [plotMatrix; [j, Acc(end)]];
%         end
%         plot(plotMatrix(:,1),plotMatrix(:,2));
%         xlabel('Mu');
%         ylabel('Accuracy');
%         hold on;
%         plotMatrix = [];
%     end
% end

% disp(Accs);
% disp(plotMatrix);
