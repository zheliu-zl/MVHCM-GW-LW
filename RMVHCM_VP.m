%% 导入数据
clear,clc;
%% Iris
% load("Iris_data.mat");
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% Mfeat
% load("Mfeat.mat");
% pos='D3';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% Webkb
% load("webkb.mat");
% pos='D6';
% truelabel{1}=Y';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% 3sources
% load("3sources.mat");
% pos='D9';
% C=max(truelabel{1});
% truelabel{1}=truelabel{1}';
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% HW2sources
% load("HW2sources.mat");
% pos='D12';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% Caltech_2
% load("Caltech_2.mat");
% pos='D15';
% data=Caltech_2.data;
% truelabel{1}=Caltech_2.Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% ORL
% load ORL.mat
%  data=X;
% truelabel{1}=Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% IS
% Data = load('IS.txt');  Y = Data(:,end); Data(:,end) = [];
% Data = mapminmax(Data',0,1); Data = Data';
% view = 2; features = [9 10];
% 
% left = 1;right=0; X=cell(1,view);
% for i=1:view
%     X{i} = Data(:, left:features(i)+right);
%     left = features(i)+right+1;
%     right = features(i)+right;
% end
% data=X;
% truelabel{1}=Y;
% H=view;
% C=max(Y);
% pos='D18';

%% BBC4
% load BBC4view_685.mat;
% C=max(truelabel{1});
% truelabel{1}=truelabel{1}';
% [~,H]=size(data);
% for h = 1 : H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% 100Leaves
load 100Leaves.mat
data=X;truelabel{1}=Y;
C=max(truelabel{1});
[~,H]=size(data);
for h=1:H
    data{h}=data{h}';
    data{h}=mapminmax(data{h},0,1);
    data{h}=data{h}';
end
pos='D21';

%%  MSRC-v1
% load MSRC-v1.mat
% data=X;
% truelabel{1}=Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end
%% forest
% Data = load('foresttype.txt');  Y = Data(:,end);
% Data = mapminmax(Data',0,1); Data = Data';Data(:,end) = Y;
% view = 2; features = [9 18];
% 
% left = 1;right=0; X=cell(1,view);
% for i=1:view
%     X{i} = Data(:, left:features(i)+right);
%     left = features(i)+right+1;
%     right = features(i)+right;
% end
% data=X;
% truelabel{1}=Y;
% C=max(truelabel{1});
% H=view;

%% MNIST-10k
% load MNIST-10k.mat
% data=X;
% truelabel{1}=Y;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% motion
% load motion.mat
% data={X_person1,X_person2};
% truelabel{1}=Y_person1';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end


%% 初始化
[n,~]=size(data{1});

times=10;
AC1=zeros(times,1);
nmi=zeros(times,1);
P=zeros(times,1);
R1=zeros(times,1);
F=zeros(times,1);
RI=zeros(times,1);
FM=zeros(times,1);
J=zeros(times,1);
metrics=[];
stdmetrics=[];
%% 开始计算
for time=1:times

    lambda=zeros(1,H);
    U=zeros(n,C);
    W=zeros(1,H);
    center = cell(1,H);
    target=0;
    for h=1:H
        lambda(h)=getBeta(data{h});
        col=size(data{h},2);
        center{h}=zeros(C,col);%C行col列
        W(h)=1;
    end
    %% 初始化聚类中心
    for h=1:H
        %center{h}=getCenter(data{h},C);
%         center{h}=fcm(data{h},C);
            [~,center{h}]=kmeans(data{h},C);
%             center{h}=get_random_center(data{h},C);
    end
    %% 开始迭代
    loop=1;
    while true
        loop=loop+1;
        distence = zeros(n,C,H);
        new_center = cell(1,H);
        %% 计算距离
        for h =1:H
            for i = 1 : n
                for j = 1 : C
                    distence(i,j,h)=alternative_metric(data{h}(i,:),center{h}(j,:),lambda(h));
                end
            end
        end
        %% 更新隶属度
        for i=1:n
            dis=zeros(1,C);
            for j=1:C
                U(i,j)=0;
                for h=1:H
                    dis(j)=dis(j)+distence(i,j,h)*W(h);
                end
            end
            [~,p]=min(dis);
            U(i,p)=1;
        end
        %% 更新类中心
        for j=1:C
            for h = 1 : H
                col=size(data{h}(i,:),2);
                fz=zeros(1,col);
                fm=0;
                for i =1 : n
                    fz=fz+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2)*data{h}(i,:);
                    fm=fm+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2);
                end
                if(fm==0)
                    new_center{h}(j,:)=center{h}(j,:);
                else
                    new_center{h}(j,:)=fz/fm;
                end
            end
        end
        center = new_center;
        %% 更新视图权重
        for h=1:H
            W(h)=1;
            for k =1:H
                fz=sum(sum(U(:,:).*distence(:,:,h)));
                fm=sum(sum(U(:,:).*distence(:,:,k)));
                W(h)=W(h)*(fz/fm);
            end
            W(h)=1/W(h)^(1/H);
        end
        %% 判断退出
        new_target=0;
        for h =1 :H
            new_target=new_target+sum(sum(U.*distence(:,:,h)*W(h)));
        end
        if(abs(new_target-target(loop-1))<1e-5)
            break;
        end
        target(loop)=new_target;
    end
    %% 标签匹配，计算正确率
    label=zeros(n,1);
    for i=1:n
        [~,p]=max(U(i,:));
        label(i)=p;
    end
    result_label=label_map(label,truelabel{1});
    Y=truelabel{1};



    AC1(time) = length(find(Y == result_label))/length(Y);
    result = CalcMeasures(Y, result_label);
    %                 Outs = valid_external(result_label,Y);
    nmi(time) = NMI(Y,result_label);
    [P(time),R1(time),F(time),RI(time),FM(time),J(time)] = Evaluate(Y,result_label);
end
meanAC=mean(AC1);
meanNMI=mean(nmi);
meanP=mean(P);
meanR1=mean(R1);
meanF=mean(F);
meanRI=mean(RI);
meanFM=mean(FM);
meanJ=mean(J);
stdAC=std(AC1);
stdNMI= std(nmi);
stdP= std(P);
stdR1= std(R1);
stdF= std(F);
stdRI= std(RI);
stdFM= std(FM);
stdJ= std(J);
metrics = [metrics;[meanAC,meanNMI,meanP,meanR1,meanF,meanRI,meanFM,meanJ]];
fprintf("ACC:%f,NMI:%f,P:%f,R:%f,F:%f,RI:%f,FM:%f,J:%f\n",meanAC,meanNMI,meanP,meanR1,meanF,meanRI,meanFM,meanJ);
stdmetrics = [stdmetrics;[stdAC,stdNMI,stdP,stdR1,stdF,stdRI,stdFM,stdJ]];


% writetable(table(M_ACC,M_EER,M_NMI,M_P,M_R,M_F,M_RI,M_FM,M_J),'聚类结果（修改后）.xlsx','WriteVariableNames',false,'Sheet','RMVHCM-VP','Range',pos);




