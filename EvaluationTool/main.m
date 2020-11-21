%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation tool boxs for "Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans" submit to Transactions on Medical Imaging2020.
%Author: Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao
%Homepage: http://dpfan.net/
%Projectpage: https://github.com/DengPingFan/Inf-Net
%First version: 2020-4-15
%Any questions please contact with dengpfan@gmail.com.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: Providing several important metrics: Dice, IoU, F1, S-m (ICCV'17), 
%          E-m (IJCAI'18), Precision, Recall, Sensitivity, Specificity, MAE.


clear all;
close all;
clc;

% ---- 1. ResultMap Path Setting ----
ResultMapPath = '../Results/';
Results = {'Lung infection segmentation','Multi-class lung infection segmentation'};

Models_LungInf = {'UNet','UNet++','Inf-Net','Semi-Inf-Net'};
Models_MultiClass_LungInf  = {'DeepLabV3Plus_Stride8','DeepLabV3Plus_Stride16','FCN8s_1100','Semi-Inf-Net_FCN8s_1100', 'Semi-Inf-Net_UNet'};

MultiClass = {'Ground-glass opacities', 'Consolidation'};

% ---- 2. Ground-truth Datasets Setting ----
DataPath = '../Dataset/TestingSet/';
Datasets = {'LungInfection-Test', 'MultiClassInfection-Test'};

% ---- 3. Evaluation Results Save Path Setting ----
ResDir = '../EvaluateResults/';

ResName='_result.txt';  % You can change the result name.

Thresholds = 1:-1/255:0;
datasetNum = length(Datasets);

for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d}   % print cur dataset name
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    if (d==1)
        curModel = Models_LungInf;
        numFolder = 1;
    else
        curModel = Models_MultiClass_LungInf;
        numFolder = 2;
    end
    
    %For Lung infection segmentation, there is only one folder;
    %For Multi-class lung infection segmentation, there are two foloders.
    for c = 1:numFolder
        modelNum = length(curModel);
        
        ResPath = [ResDir dataset '-' int2str(c) '-mat/']; % The result will be saved in *.mat file so that you can used it for the next time.
        if ~exist(ResPath,'dir')
            mkdir(ResPath);
        end
        
        resTxt = [ResDir dataset '-' int2str(c) ResName];  % The evaluation result will be saved in `../Resluts/Result-XXXX` folder.
        fileID = fopen(resTxt,'w');
        
        
        for m = 1:modelNum
            model = curModel{m}   % print cur model name
            
            if (d==1)
                gtPath = [DataPath dataset '/GT/'];
                resMapPath = [ResultMapPath Results{d}, '/' model '/'];
            else
                gtPath = [DataPath dataset '/GT' '-' int2str(c) '/'];
                resMapPath = [ResultMapPath Results{d}, '/' MultiClass{c}, '/' model '/'];
            end
            
            imgFiles = dir([resMapPath '*.png']);
            imgNUM = length(imgFiles);
            
            [threshold_Emeasure, threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
            [threshold_Sensitivity, threshold_Specificity, threshold_Dice] = deal(zeros(imgNUM,length(Thresholds)));
            
            [Smeasure, MAE] =deal(zeros(1,imgNUM));
            
            for i = 1:imgNUM
                name =  imgFiles(i).name;
                fprintf('Evaluating(%s Dataset,%s Model, %s Image): %d/%d\n',dataset, model, name, i,imgNUM);
                
                %load gt
                gt = imread([gtPath name]);
                
                if (ndims(gt)>2)
                    gt = rgb2gray(gt);
                end
                
                if ~islogical(gt)
                    gt = gt(:,:,1) > 128;
                end
                
                %load resMap
                resmap  = imread([resMapPath name]);
                
                %check size
                if size(resmap, 1) ~= size(gt, 1) || size(resmap, 2) ~= size(gt, 2)
                    resmap = imresize(resmap,size(gt));
                    imwrite(resmap,[resMapPath name]);
                    fprintf('Resizing have been operated!! The resmap size is not math with gt in the path: %s!!!\n', [resMapPath name]);
                end
                
                resmap = im2double(resmap(:,:,1));
                
                %normalize resmap to [0, 1]
                resmap = reshape(mapminmax(resmap(:)',0,1),size(resmap));
                
                % S-meaure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
                Smeasure(i) = StructureMeasure(resmap,logical(gt));
                   
                [threshold_E, threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
                [threshold_Spe, threshold_Dic]  = deal(zeros(1,length(Thresholds)));
                for t = 1:length(Thresholds)
                    threshold = Thresholds(t);
                    [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), threshold_Dic(t), ~] = Fmeasure_calu(resmap,double(gt),size(gt),threshold);
                    
                    Bi_resmap = zeros(size(resmap));
                    Bi_resmap(resmap>=threshold)=1;
                    threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
                end
                
                threshold_Emeasure(i,:) = threshold_E;
                threshold_Sensitivity(i,:) = threshold_Rec;
                threshold_Specificity(i,:) = threshold_Spe;
                threshold_Dice(i,:) = threshold_Dic;
                
                MAE(i) = mean2(abs(double(logical(gt)) - resmap));
                
            end
            
            mae = mean2(MAE);
            
            %Sensitivity
            column_Sen = mean(threshold_Sensitivity,1);
            meanSen = mean(column_Sen);
            maxSen = max(column_Sen);
            
            %,Specificity
            column_Spe = mean(threshold_Specificity,1);
            meanSpe = mean(column_Spe);
            maxSpe = max(column_Spe);
            
            %Dice
            column_Dic = mean(threshold_Dice,1);
            meanDic = mean(column_Dic);
            maxDic = max(column_Dic);
            
            %E-m
            column_E = mean(threshold_Emeasure,1);
            meanEm = mean(column_E);
            maxEm = max(column_E);
            
            %Sm
            Sm = mean2(Smeasure);
               
            save([ResPath model],'Sm', 'mae', 'column_Dic', 'column_Sen', 'column_Spe', 'column_E','maxDic','maxEm','maxSen','maxSpe','meanDic','meanEm','meanSen','meanSpe');
            fprintf(fileID, '(Dataset:%s; Model:%s) meanDic:%.3f;meanSen:%.3f;meanSpe:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f.\n',dataset,model,meanDic,meanSen,meanSpe,Sm,meanEm,mae);
            fprintf('(Dataset:%s; Model:%s) meanDic:%.3f;meanSen:%.3f;meanSpe:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f.\n',dataset,model,meanDic,meanSen,meanSpe,Sm,meanEm,mae);
        end
    end
    toc;
    
end


