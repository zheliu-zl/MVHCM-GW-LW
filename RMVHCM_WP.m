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
%     %data{h}=data{h}';
%     data{h}=mapminmax(data{h},0,1);
%     %data{h}=data{h}';
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

%% BBC
% load BBC.mat;
% %data={data1,data2,data3,data4};
% truelabel{1}=truelabel{1}';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% BBCSport
% load BBCSport.mat
% truelabel{1}=truelabel{1}';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% prokaryotic
% load prokaryotic.mat
% data={gene_repert,proteome_comp,text};
% truelabel{1}=truth;
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%         data{h}=data{h}';
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
    loop=1;
    lambda=zeros(1,H);
    U=zeros(n,C);
    W=ones(C,H);
    center = cell(1,H);
    target=0;
    for h=1:H
        lambda(h)=getBeta(data{h});
        col=size(data{h},2);
        center{h}=zeros(C,col);
    end
    %% 初始化聚类中心
    for h=1:H
%                 center{h}=fcm(data{h},C);
%                          [~,center{h}]=kmeans(data{h},C);
        %center{h}=Centroids_Initialization(data{h},C);
        center{h}=get_random_center(data{h},C);
    end
    %% 开始迭代
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
                    dis(j)=dis(j)+distence(i,j,h)*W(j,h);
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
                    fm=1e-12;
                end
                new_center{h}(j,:)=fz/fm;
            end
        end
        center = new_center;
        %% 更新视图权重
        for j = 1:C
            for h=1:H
                W(j,h)=1;
                for k =1:H
                    fz=sum(U(:,j).*distence(:,j,h));
                    fm=sum(U(:,j).*distence(:,j,k));
                    if(fz==0)
                        fz=1e-12;
                    end
                    if(fm==0)
                        fm=1e-12;
                    end
                    W(j,h)=W(j,h)*(fz/fm);% 分母有可能为0
                end
                W(j,h)=1/W(j,h)^(1/H);
            end
        end
        %% 判断退出
        new_target=0;
        for h =1 :H
            for j=1:C
                new_target=new_target+sum(W(j,h)*U(:,j).*distence(:,j,h));
            end
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


