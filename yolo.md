* YOLO 
   * 장점
      * 매우 알고리즘이 심플
      * 매우 빠르다.
      * background에 대한 error가 fast rcnn보다 상대적으로 적다
   * 단점
      * 빠른대신 감지율이 낮다(ver2에선 이를 해결)
      * 작은 오브젝트에 대해서는 검출이 잘 안된다.
* 알고리즘
   * faster rcnn류와는 다르게 box, class관련에서 따로 보지 않고(rpn), 한개의 문제로 봄.
      * single regression problem
   * 즉, 하나의 network상에서 box에 대한 class를 계산
* 세부 알고리즘

![image](https://cloud.githubusercontent.com/assets/6295576/25041544/c33e7c22-214b-11e7-8479-f9b95b70d72a.png)

  * 위의 보시는거 와 같이, 하나의 network에서 모든걸 판단
  * 마지막 layer만 다르고 일반 cnn과 같음. -> 신기함.
  * last layer에서, 
     * 7x7 cell
     * B Bounding Box (B=2) :  x, y, w, h (x B) 
        * x, y 의 기준은 각 Cell 하나의 크기가 1이고,  그 각 하나의 Cell 기준으로 convert된 단위 = 0~1
        * w, h 는 이미지 사이즈로 normalized 된 scale = 0~1
     * Bounding Box에 대한 confidence : p(x B)
        * Pr(Object)*IOU([개념](https://oss.navercorp.com/chullhwan-song/Study/issues/16))   
           * grid cell의 B개의 bounding box(bbox)가 존재
           * bbox안에 object가 졵재하지 않는 다면,  0
           * IOU : predict box와 ground truth box 사이의 겹침여부 계산
     * predefined class prob(C=20) : c1, c2,,,,c20
         * 각 Cell에서는 클래스(C)의 confidence를 갖는다. 한 Cell마다 C=20개 갖음.
         * Pr(class|Objects)
      * test time에서, 위의 두개의 multiply
         * class-specific-confidence score =  Pr(class|Objects) x  Pr(Object)*IOU    
          * 아래의 infer1~4의 그림은 이를 설명하고 있다. 

infer-1
![image](https://cloud.githubusercontent.com/assets/6295576/25041702/c4b7ecf4-214c-11e7-9892-c62808244676.png)

* 이 그림에서, 위에서 설명하거와 같이, 7x7x30 의 의미는
    * 7x7 = grid
    * 5 = (x, y, w, h, p)   
       * B가 2이니, (x, y, w, h, p)x2 = 10
       * 노란색 Box
    * class = 20
    * so, 7x7x(5x2+20) = 7x7x30      

infer-2
![image](https://cloud.githubusercontent.com/assets/6295576/25060779/8c68804a-21e0-11e7-9e60-871975e4a996.png)
infer-3     
![image](https://cloud.githubusercontent.com/assets/6295576/25060782/a3a912ce-21e0-11e7-8068-c0c9d3e36ec2.png)
infer-4
![image](https://cloud.githubusercontent.com/assets/6295576/25060785/ab7915da-21e0-11e7-948d-990ac55babc6.png)
infer-5 : 
![image](https://cloud.githubusercontent.com/assets/6295576/25060814/504c460e-21e1-11e7-84d5-23a3ba8a9bf0.png)

* paper 2.3장 Inference = infer-5
   * 최종적으로 98(7x7x2)box를 얻음. -> nmx
  
* training
   * 448x448 input
   * last layer에서 class prob & bbox 예측
   * box 정보 scale은 위에서 설명
   * leaky rectified linear activation 
   * 학습은 최종 loss function이 그 학습에서 핵심, 이외는 일반 cnn과 다를바 없음.

![image](https://cloud.githubusercontent.com/assets/6295576/25060873/da0fe228-21e2-11e7-98cb-8d2a9f10746c.png)

   *  Parameter 설명
       * ![image](https://cloud.githubusercontent.com/assets/6295576/25077040/4a22b71e-2361-11e7-96b0-80210ca93e48.png)  : 오브젝트가 포함되어 있을경우의 confidence = 5
       * ![image](https://cloud.githubusercontent.com/assets/6295576/25077043/4d9188d0-2361-11e7-89a1-4e01325bee9d.png) : 오브젝트가 포함되어 있지 않을경우의 confidence = 0.5
       *  ![image](https://cloud.githubusercontent.com/assets/6295576/25077127/ecb5ca20-2361-11e7-8a49-6b8946472250.png) : 오브젝트가 그 셀에 존재 = 0 or 1
       *  ![image](https://cloud.githubusercontent.com/assets/6295576/25077139/00d998ec-2362-11e7-844a-4ed5c0bccfca.png) : 오브젝트가 그 셀에 존재하지 않을때 = 0 or 1
    * training 
       * dropout = 0.5
       * data argumentation
         * random scaling & translation of up to 20% original size

* result
  * 속도
![image](https://media.oss.navercorp.com/user/1196/files/b62821f0-5521-11e7-9c7c-d1606c6a6028)
 * 정확도
![image](https://media.oss.navercorp.com/user/1196/files/d8cf5d04-5521-11e7-8c71-f07a43adec2c)

* 이후 2016년 말쯤~ Yolo2 
    * Batch Normalization, High Resolution Classifier(input size의 크기 224x224->448x448), Convolutional With Anchor Boxes( faster rcnn 차용) 등등사용하여, SSD나 faster rcnn 성능에 비등.  속도는 좀 느려짐(하지만, 기존 localization 알고리즘보다는 빠름)


### Ref
 * 이 두 Ref[1][2]을 너무 많이 참조 되었음.:)
   * [1]  https://www.youtube.com/watch?v=L0tzmv--CGY&feature=youtu.be 
   * [2] https://curt-park.github.io/2017-03-26/yolo/
    


