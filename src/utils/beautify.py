"""
Beautify Images Utils
"""

def get_top_frames(scores, num, fps, dispersed = True):
  '''
  Returns list of indexes for number frames with the highest scores as 
  specified by the user. 

  Users can define the 'dispersed' function if they wish to have num images
  taken from different parts of the video. In this instance, we randomly sample 
  10% of the frames from the video and score these frames.
  
  Otherwise the function just returns the best num images from the frames scored.
  '''
  if len(scores)<=1000:
    dispersed = False
    
  if dispersed:
    
    tmp = []

    while True:
      if len(tmp) == int(0.1*len(scores)):
        break   
      
      sampled_frame = random.choice(scores)

      if len(tmp)==0:
        tmp.append(sampled_frame)
      else:
        flag = False
        
        for i in tmp:
          if i-fps<=sampled_frame<=i+fps:
            flag = True
          break
        
        if flag == False:
          tmp.append(sampled_frame)
    
    idx = sorted(list(zip(*heapq.nlargest(num, enumerate(tmp), 
                                           key = operator.itemgetter(1))))[0])
    
    return sorted([scores.index(j) for j in [tmp[i] for i in idx]])

  else:
    return sorted(list(zip(*heapq.nlargest(num, enumerate(scores), 
                                           key = operator.itemgetter(1))))[0])


def beautify():
    pass
