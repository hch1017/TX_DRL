from turtle import distance, speed
from AI_Interface import *
from math import asin, acos, sin, cos, exp, pi
import numpy as np

# class Axis():
#     def __init__(self, a = 0, b = 0, c = 0):#a,b,c依次是偏航角，俯仰角，滚转角
#         GxOy = np.array([[cos((a+pi/2)),  -sin((a+pi/2)),   0        ],
#                          [sin((a+pi/2)), cos((a+pi/2))  ,   0        ],
#                          [0            ,  0             ,   1        ]])
#         GzOx = np.array([[cos(b)       ,  0             ,   -sin(b)  ],
#                          [0            ,  1             ,   0        ],
#                          [sin(b)       ,  0             ,   cos(b)   ]])
#         GyOz = np.array([[1            ,  0             ,   0        ],
#                          [0            ,  -cos(c)       ,   sin(c)   ],
#                          [0            ,  -sin(c)       ,   -cos(c)  ]])
#         self.base = GxOy @ GzOx @ GyOz
#     def transform(self, component, reverse = False):
#         if reverse:#从机体坐标系变换到北天东坐标系
#             return self.base @ component
#         else:#从北天东坐标系变换到机体坐标系
#             return self.base.T @ component

class Axis():
    def __init__(self, a = 0, b = 0, c = 0):#a,b,c依次是偏航角，俯仰角，滚转角
        a = a+pi/2
        b = -b
        c = -c
        e0, e1, e2, e3 = cos(a/2)*cos(b/2)*cos(c/2)+sin(a/2)*sin(b/2)*sin(c/2),\
                         cos(a/2)*cos(b/2)*sin(c/2)-sin(a/2)*sin(b/2)*cos(c/2),\
                         cos(a/2)*sin(b/2)*cos(c/2)+sin(a/2)*cos(b/2)*sin(c/2),\
                         sin(a/2)*cos(b/2)*cos(c/2)-cos(a/2)*sin(b/2)*sin(c/2)
        self.base = np.array([[e0**2+e1**2-e2**2-e3**2, 2*(e1*e2+e0*e3)        , 2*(e1*e3-e0*e2)        ],
                              [2*(e1*e2-e0*e3)        , e0**2-e1**2+e2**2-e3**2, 2*(e2*e3+e0*e1)        ],
                              [2*(e0*e2+e1*e3)        , 2*(e2*e3-e0*e1)        , e0**2-e1**2-e2**2+e3**2]])
    def transform(self, component, reverse = False):
        if reverse:#从机体坐标系变换到北天东坐标系
            return self.base.T @ component
        else:#从北天东坐标系变换到机体坐标系
            return self.base @ component

def getDistance(nextRedInput, nextBlueInput):
    # distance = ((nextRedInput.m_AircraftMoveInfo.m_dSelfLon - nextBlueInput.m_AircraftMoveInfo.m_dSelfLon)**2 +\
    #             (nextRedInput.m_AircraftMoveInfo.m_dSelfLat - nextBlueInput.m_AircraftMoveInfo.m_dSelfLat)**2 +\
    #             (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt - nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt)**2)**0.5
    distance = nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtDis
    return distance

def getSpeed(Input):
    speed = ((Input.m_AircraftMoveInfo.m_fVN)**2 +\
            (Input.m_AircraftMoveInfo.m_fVU)**2 +\
            (Input.m_AircraftMoveInfo.m_fVE)**2)**0.5
    return speed

def getAngleAdvantage(nextRedInput, nextBlueInput, versionID):
    # distance = getDistance(nextRedInput, nextBlueInput)
    redSpeed = getSpeed(nextRedInput)
    # blueSpeed = getSpeed(nextBlueInput)
    redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
                   nextRedInput.m_AircraftMoveInfo.m_fPitch,
                   nextRedInput.m_AircraftMoveInfo.m_fRoll)
    # blueAxis = Axis(nextBlueInput.m_AircraftMoveInfo.m_fYaw,
    #                 nextBlueInput.m_AircraftMoveInfo.m_fPitch,
    #                 nextBlueInput.m_AircraftMoveInfo.m_fRoll)
    if versionID == 0:
        # positionVector = np.array([nextBlueInput.m_AircraftMoveInfo.m_dSelfLon - nextRedInput.m_AircraftMoveInfo.m_dSelfLon,
        #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfLat - nextRedInput.m_AircraftMoveInfo.m_dSelfLat,
        #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt - nextRedInput.m_AircraftMoveInfo.m_dSelfAlt])
        # transformedPositionVectorRed = redAxis.transform(positionVector)
        # redSightAngle = acos(transformedPositionVectorRed[0]/(distance+1e-4))
        # transformedPositionVectorBlue = blueAxis.transform(-positionVector)
        # blueSightAngle = acos(transformedPositionVectorBlue[0]/(distance+1e-4))
        redSightAngle = acos(cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180))
        blueSightAngle = acos(cos(nextBlueInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * cos(nextBlueInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180))
        # reward = (blueSightAngle - redSightAngle)/(2*pi)+0.5
        reward = (blueSightAngle - redSightAngle)/(0.01)
    elif versionID == 1 :
        transformedPositionVectorRed = np.array([cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180),
                                                 cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * sin(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180),
                                                 sin(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180)])
        positionVector = redAxis.transform(transformedPositionVectorRed, reverse=True)
        phiRed = acos((nextRedInput.m_AircraftMoveInfo.m_fVE * positionVector[0] +\
                       nextRedInput.m_AircraftMoveInfo.m_fVU * positionVector[1] +\
                       nextRedInput.m_AircraftMoveInfo.m_fVN * positionVector[2])/(redSpeed+1e-4))
        # phiRed = acos((nextRedInput.m_AircraftMoveInfo.m_fVE * (nextBlueInput.m_AircraftMoveInfo.m_dSelfLon - nextRedInput.m_AircraftMoveInfo.m_dSelfLon) +\
        #                nextRedInput.m_AircraftMoveInfo.m_fVU * (nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt - nextRedInput.m_AircraftMoveInfo.m_dSelfAlt) +\
        #                nextRedInput.m_AircraftMoveInfo.m_fVN * (nextBlueInput.m_AircraftMoveInfo.m_dSelfLat - nextRedInput.m_AircraftMoveInfo.m_dSelfLat))/(redSpeed*distance+1e-4))
        # phiBlue = acos((nextBlueInput.m_AircraftMoveInfo.m_fVE * (nextBlueInput.m_AircraftMoveInfo.m_dSelfLon - nextRedInput.m_AircraftMoveInfo.m_dSelfLon) +\
        #                 nextBlueInput.m_AircraftMoveInfo.m_fVU * (nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt - nextRedInput.m_AircraftMoveInfo.m_dSelfAlt) +\
        #                 nextBlueInput.m_AircraftMoveInfo.m_fVN * (nextBlueInput.m_AircraftMoveInfo.m_dSelfLat - nextRedInput.m_AircraftMoveInfo.m_dSelfLat))/(blueSpeed*distance+1e-4))
        reward = 1 - abs(phiRed)/pi
    return reward

def getDistanceAdvantage(nextRedInput, currentRedInput, versionID):
    # distance = getDistance(nextRedInput, nextBlueInput)
    distance = nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtDis
    pre_distance = currentRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtDis
    if versionID == 1:
        optimalDistance = 200
        D0 = 2000 #D0用于调整奖励梯度
        return exp(-abs(distance - optimalDistance)/D0)
    if versionID == 2:
        if distance > 600:
            return -(distance -  600)/ 10000
        else:
            return -1
    if versionID == 3:
        if distance == 0 or distance < 600:
            return -1
        elif distance > 600:
            if distance-pre_distance <= 0:
                return 1
            else:
                return -1

def getAltitudeAdvantage(nextRedInput, nextBlueInput, currentRedInput, versionID):
    reward = 0
    if versionID == 1:
        optimalAlt = 6000
        if nextRedInput.m_AircraftMoveInfo.m_dSelfAlt < nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt:
            reward = nextRedInput.m_AircraftMoveInfo.m_dSelfAlt / (nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt+1e-4) - 0.5
        else:
            reward = exp(-abs(nextRedInput.m_AircraftMoveInfo.m_dSelfAlt - optimalAlt)/(nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt+1e-4)) #最佳空战高度6000
    elif versionID == 2:
        Delta_h = nextRedInput.m_AircraftMoveInfo.m_dSelfAlt - nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt
        if Delta_h < -5000:
            return -1
        elif Delta_h < 5000:
            return -1 + (Delta_h+5000)/5000
        else:
            return -0.1
    elif versionID == 3:
        if (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt < currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) and \
                (currentRedInput.m_AircraftMoveInfo.m_dSelfAlt<1000):
            return -2
        elif (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt > currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) and (currentRedInput.m_AircraftMoveInfo.m_dSelfAlt<1000):
            return 2
        elif nextRedInput.m_AircraftMoveInfo.m_dSelfAlt > currentRedInput.m_AircraftMoveInfo.m_dSelfAlt and currentRedInput.m_AircraftMoveInfo.m_dSelfAlt>1000:
            return 0.1
        elif nextRedInput.m_AircraftMoveInfo.m_dSelfAlt < currentRedInput.m_AircraftMoveInfo.m_dSelfAlt and currentRedInput.m_AircraftMoveInfo.m_dSelfAlt>1000:
            return -0.1
    elif versionID == 4:
        if (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt < currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) and (currentRedInput.m_AircraftMoveInfo.m_dSelfAlt<1000):
            return -5
        # elif (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt > currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) and (currentRedInput.m_AircraftMoveInfo.m_dSelfAlt<1000):
        #     return 5
    return reward

def getSpeedAdvantage(nextRedInput, nextBlueInput, versionID):
    if versionID == 1:
        redSpeed = getSpeed(nextRedInput)
        blueSpeed = getSpeed(nextBlueInput)
        optimalSpeed = 200
        if 1.5*blueSpeed < optimalSpeed: #最佳空战速度200
            if redSpeed > optimalSpeed:
                reward = exp((optimalSpeed - redSpeed)/optimalSpeed)
            elif 1.5*blueSpeed < redSpeed:
                reward = 1
            elif 0.5*blueSpeed < redSpeed:
                reward = redSpeed/(blueSpeed+1e-4) -0.5
            else:
                reward = 0.1
        else:
            if redSpeed > optimalSpeed:
                reward = exp((optimalSpeed-redSpeed)/optimalSpeed)
            elif 0.5*blueSpeed < redSpeed and 1.5*blueSpeed >= redSpeed:
                reward = 0.4*(redSpeed/optimalSpeed -redSpeed/(blueSpeed+1e-4))
            elif 0.5*blueSpeed >= redSpeed:
                reward = 0.1
        return reward
    if versionID == 2:
        redSpeed = getSpeed(nextRedInput)
        blueSpeed = getSpeed(nextBlueInput)
        if blueSpeed < 0.6 * redSpeed:
            return -0.1
        elif blueSpeed < 1.5 * redSpeed:
            return 0.5 - blueSpeed / redSpeed
        else:
            return -1.0

def SRadarDetectRange(nextRedInput, nextBlueInput, nextRedOutput, nextBlueOutput):
    # distance = getDistance(nextRedInput, nextBlueInput)
    # speed = getSpeed(nextRedInput)

    # if not nextRedInput.m_FCInfo.m_eAircraftMainState:  # 超视距
    #     speedPitch = asin(nextRedInput.m_AircraftMoveInfo.m_fVU/(speed+1e-4))
    #     redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
    #                    speedPitch,
    #                    nextRedInput.m_AircraftMoveInfo.m_fRoll)
    # else:
    #     redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
    #                    nextRedInput.m_AircraftMoveInfo.m_fPitch,
    #                    nextRedInput.m_AircraftMoveInfo.m_fRoll)

    # positionVector = np.array([nextBlueInput.m_AircraftMoveInfo.m_dSelfLon -nextRedInput.m_AircraftMoveInfo.m_dSelfLon,
    #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfLat -nextRedInput.m_AircraftMoveInfo.m_dSelfLat,
    #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt -nextRedInput.m_AircraftMoveInfo.m_dSelfAlt])
    # transformedPositionVectorRed = redAxis.transform(positionVector)

    # positionEle = asin(transformedPositionVectorRed[2]/(distance+1e-4))*180/pi
    # positionAzi = acos(transformedPositionVectorRed[0]/(distance*abs(cos(positionEle))+1e-4))*180/pi
    positionEle = nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC
    positionAzi = nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC

    if nextRedInput.m_FCInfo.m_eAircraftMainState:  # 视距内
        if positionAzi > -5 and positionAzi < 5 and \
            positionEle >= -7 and positionEle <= 53 :
            return True
        else:
            return False
    else:
        if positionAzi  > nextRedOutput.m_RadarCtrlCmd.m_fAziScanCent - nextRedOutput.m_RadarCtrlCmd.m_eAziScanRange/2 and\
            positionAzi < nextRedOutput.m_RadarCtrlCmd.m_fAziScanCent + nextRedOutput.m_RadarCtrlCmd.m_eAziScanRange/2 and\
            positionEle > nextRedOutput.m_RadarCtrlCmd.m_fEleScanCent - 4.5 * nextRedOutput.m_RadarCtrlCmd.m_eEleScanLine/2 and\
            positionEle < nextRedOutput.m_RadarCtrlCmd.m_fEleScanCent + 4.5 * nextRedOutput.m_RadarCtrlCmd.m_eEleScanLine/2 :
            return True
        else:
            return False

def SMRAAMCaptured(nextRedInput):
    for i in range(int(nextRedInput.m_MRAAMDataMsgSet.m_uiMsgNum)):
        if nextRedInput.m_MRAAMDataMsgSet.m_MRAAMDataMsg[i].m_bCapture:
            return True
    return False

def SMRAAMCapturedInRange(nextRedInput, nextBlueInput):
    distance = getDistance(nextRedInput, nextBlueInput)
    if (SMRAAMCaptured(nextRedInput)):
        if (distance > nextRedInput.m_FCInfo.m_fRmin and \
            distance < nextRedInput.m_FCInfo.m_fRmax):
            if (distance < nextRedInput.m_FCInfo.m_fRnoescape):
                return 0.8
            else:
                return 0.55
        else:
            return 0.3
    else:
        return 0

def SRadarLocked(nextRedInput, nextBlueInput):
    # maxRadarLockedTolerence = 10000 #人为设定的雷达误差
    # minRadarLockedDistance = 100000
    speed = getSpeed(nextRedInput)

    redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
                   nextRedInput.m_AircraftMoveInfo.m_fPitch,
                   nextRedInput.m_AircraftMoveInfo.m_fRoll)
    transformedPositionVectorRed = np.array([cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180),
                                             cos(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180) * sin(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtAziC*pi/180),
                                             sin(nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtEleC*pi/180)])
    positionVector = redAxis.transform(transformedPositionVectorRed, reverse=True)


    if not nextRedInput.m_FCInfo.m_eAircraftMainState:  # 视距外
        speedPitch = asin(nextRedInput.m_AircraftMoveInfo.m_fVU/(speed+1e-4))
        redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
                       speedPitch,
                       nextRedInput.m_AircraftMoveInfo.m_fRoll)
    # else:
    #     redAxis = Axis(nextRedInput.m_AircraftMoveInfo.m_fYaw,
    #                    nextRedInput.m_AircraftMoveInfo.m_fPitch,
    #                    nextRedInput.m_AircraftMoveInfo.m_fRoll)
    # positionVector = np.array([nextBlueInput.m_AircraftMoveInfo.m_dSelfLon -nextRedInput.m_AircraftMoveInfo.m_dSelfLon,
    #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfLat -nextRedInput.m_AircraftMoveInfo.m_dSelfLat,
    #                            nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt -nextRedInput.m_AircraftMoveInfo.m_dSelfAlt])
    transformedPositionVectorRed = redAxis.transform(positionVector)

    # for i in range(int(nextRedInput.m_RadarInfo.m_uiTgtNum)):
        # SRadarLockedPositionAlt = nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtDis * \
        #                           sin(nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtEle*180/pi)
        # projectedDistance = nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtDis * \
        #                     cos(nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtEle*180/pi)
        # SRadarLockedPositionLon = projectedDistance * \
        #                           cos(nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtAzi*180/pi)
        # SRadarLockedPositionLat = projectedDistance * \
        #                           sin(nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtAzi*180/pi)
        # SRadarLockedDistance = (((SRadarLockedPositionLon - transformedPositionVectorRed[0])**2) + \
        #                         ((SRadarLockedPositionLat - transformedPositionVectorRed[1])**2) + \
        #                         ((SRadarLockedPositionAlt - transformedPositionVectorRed[2])**2))**0.5

    #     SRadarLockedDistance = abs(nextRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtDis - nextRedInput.m_HMDInfo.m_TgtInfo[0].m_fTgtDis)
    #     if SRadarLockedDistance < minRadarLockedDistance:
    #         minRadarLockedDistance = SRadarLockedDistance
    # if minRadarLockedDistance < maxRadarLockedTolerence:
    #     return True
    # else:
    #     return False
    if int(nextRedInput.m_RadarInfo.m_uiTgtNum) > 0:
        return True
    else:
        return False

def AAMHit(nextRedInput):
    for i in range(int(nextRedInput.m_AAMDataSet.m_iAAMNum)):
        if nextRedInput.m_AAMDataSet.m_AAMData[i].m_fTgtDis < 30 and \
            nextRedInput.m_AAMDataSet.m_AAMData[i].m_fTgtDis > 0.001:
            if nextRedInput.m_AAMDataSet.m_AAMData[i].m_uiPlaneID == nextRedInput.m_AircraftBasicInfo.m_ID:
                return 2
            else:
                return -2
    return 0

def getStabilityReward(nextRedInput, currentRedOutput):
    from scipy.interpolate import interpn
    import numpy as np
    if currentRedOutput.m_FlyCtrlCmd.m_fThrottle >= 0.62 and currentRedOutput.m_FlyCtrlCmd.m_fThrottle <= 0.65:
        throttleType = 1
    elif currentRedOutput.m_FlyCtrlCmd.m_fThrottle > 0.65 and currentRedOutput.m_FlyCtrlCmd.m_fThrottle < 1:
        throttleType = 2
    elif currentRedOutput.m_FlyCtrlCmd.m_fThrottle == 1:
        throttleType = 3
    else:
        throttleType = 0

    if nextRedInput.m_AircraftMoveInfo.m_fMach < 0.3 or nextRedInput.m_AircraftMoveInfo.m_fMach > 1.9\
    or nextRedInput.m_AircraftMoveInfo.m_dSelfAlt < 1000 or nextRedInput.m_AircraftMoveInfo.m_dSelfAlt > 15000:
        return -1

    altIndex = np.array([1000.,3000.,5000.,7000.,9000.,11000.,13000.,15000.])
    # machIndexType1 = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    machIndexType3 = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
    type1NormalAccelPre = np.array([[1.267,0    ,0    ,0    ,0    ,0    ,0    ,0    ], 
                                    [2.186,1.724,1.357,1.056,0    ,0    ,0    ,0    ], 
                                    [3.208,2.537,2.006,1.570,1.218,0    ,0    ,0    ], 
                                    [4.188,3.332,2.658,2.098,1.643,1.263,0    ,0    ], 
                                    [5.053,4.031,3.215,2.546,2.005,1.573,1.231,0    ], 
                                    [5.865,4.678,3.737,2.976,2.409,1.903,1.486,1.156], 
                                    [6.536,5.230,4.174,3.367,2.800,2.227,1.738,1.334]])
    type3NormalAccel = np.array([[2.662,2.163,1.723,1.338,0    ,0    ,0    ,0    ], 
                                 [3.946,3.163,2.523,1.994,1.576,1.206,0    ,0    ], 
                                 [5.365,4.338,3.491,2.763,2.146,1.623,1.204,0    ], 
                                 [6.893,5.559,4.464,3.545,2.779,2.118,1.578,1.148], 
                                 [8.239,6.758,5.514,4.393,3.437,2.616,1.948,1.417], 
                                 [9.000,7.865,6.442,5.202,4.139,3.153,2.347,1.707], 
                                 [9.000,8.421,7.003,5.725,4.599,3.557,2.676,1.943], 
                                 [6.650,7.082,6.416,5.523,4.606,3.642,2.767,2.005], 
                                 [0    ,6.787,6.790,6.003,5.068,4.031,3.072,2.225], 
                                 [0    ,0    ,6.787,6.351,5.481,4.392,3.360,2.432], 
                                 [0    ,0    ,3.188,6.567,5.877,4.751,3.653,2.646], 
                                 [0    ,0    ,0    ,6.276,6.140,5.073,3.905,2.825], 
                                 [0    ,0    ,0    ,5.137,6.225,5.309,4.146,2.991], 
                                 [0    ,0    ,0    ,0    ,5.784,5.410,4.312,3.111], 
                                 [0    ,0    ,0    ,0    ,4.825,5.392,4.411,3.165], 
                                 [0    ,0    ,0    ,0    ,0    ,2.655,4.354,3.076], 
                                 [0    ,0    ,0    ,0    ,0    ,0    ,2.662,1.482]])
    zeroNormalAccel = np.zeros_like(type3NormalAccel)
    type1NormalAccel = np.zeros_like(type3NormalAccel)
    m, n = type1NormalAccelPre.shape
    type1NormalAccel[:m,:] = type1NormalAccelPre
    table = np.array([zeroNormalAccel, type1NormalAccel, type3NormalAccel])
    index = (np.array([0.0, 0.635, 1.0]), machIndexType3, altIndex)
    query = np.array([currentRedOutput.m_FlyCtrlCmd.m_fThrottle,
                      nextRedInput.m_AircraftMoveInfo.m_fMach,
                      nextRedInput.m_AircraftMoveInfo.m_dSelfAlt])

    value = interpn(index, table, query, method='linear')

    if value[0] <= 0:
        return -1
    elif nextRedInput.m_AircraftMoveInfo.m_fNormalAccel > value:
        # return -(value - nextRedInput.m_AircraftMoveInfo.m_fNormalAccel)**2
        return -1
    else:
        return 0

def getIncidentReward(currentRedInput,
                      currentBlueInput,
                      nextRedInput,
                      nextBlueInput):
    if (currentRedInput.m_AircraftBasicInfo.m_bAlive and \
        (not nextRedInput.m_AircraftBasicInfo.m_bAlive)):
        return -1#被击毁
    if (currentRedInput.m_AircraftBasicInfo.m_fFuel > 0 and \
        nextRedInput.m_AircraftBasicInfo.m_fFuel <=0):
        return -1#油尽
    if (nextRedInput.m_AircraftMoveInfo.m_dSelfAlt <= 500):
        return -1#撞地
    if (currentBlueInput.m_AircraftBasicInfo.m_bAlive and \
        (not nextBlueInput.m_AircraftBasicInfo.m_bAlive)):
        return 1#击毁敌机
    if (currentBlueInput.m_AircraftBasicInfo.m_fFuel > 0 and \
        nextBlueInput.m_AircraftBasicInfo.m_fFuel <=0):
        return 1#敌机油尽
    # if (nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt <= 500):
    #     return 1#敌机撞地
    # if (currentRedInput.m_AircraftBasicInfo.m_bAlive and \
    #     (not nextRedInput.m_AircraftBasicInfo.m_bAlive) and \
    #     currentBlueInput.m_AircraftBasicInfo.m_bAlive and \
    #     (not nextBlueInput.m_AircraftBasicInfo.m_bAlive)):
    #     return 0#同归于尽
    else:
        return 0

def getReward(currentRedInput,
              currentBlueInput,
              currentRedOutput,
              currentBlueOutput,
              nextRedInput,
              nextBlueInput):
    reward = 0
    reward += getIncidentReward(currentRedInput,
                                currentBlueInput,
                                nextRedInput,
                                nextBlueInput)

    # if (SRadarDetected(nextRedInput, nextBlueInput, currentRedOutput, currentBlueOutput)):
    #     reward += 0.05
    # if (SRadarDetected(nextBlueInput, nextRedInput, currentBlueOutput, currentRedOutput)):
    #     reward -= 0.05

    # if (SRadarLocked(nextRedInput, nextBlueInput)):
    #     reward += 0.2#火控雷达探测到信息
    # if (SRadarLocked(nextBlueInput, nextRedInput)):
    #     reward -= 0.2#被火控雷达探测到信息



    # reward += SMRAAMCapturedInRange(nextBlueInput, nextRedInput)
    reward += 0.1 * getAngleAdvantage(nextRedInput, nextBlueInput, 0)
    # reward += 0.5 * getSpeedAdvantage(nextRedInput, nextBlueInput, 1)
    # reward += 0.001 * getDistanceAdvantage(nextRedInput, nextBlueInput, currentRedInput, 3)
    reward += 0.1 * getAltitudeAdvantage(nextRedInput, nextBlueInput,currentRedInput, 3)
    reward += 0.1 * AAMHit(nextRedInput)
    #reward += 0.05 * getStabilityReward(nextRedInput, currentRedOutput)
    return reward

if __name__ == '__main__':
    currentRedInput = AIPilotInput()
    currentBlueInput = AIPilotInput()
    currentRedOutput = AIPilotOutput()
    currentBlueOutput = AIPilotOutput()
    nextRedInput = AIPilotInput()
    nextBlueInput = AIPilotInput()

    nextBlueInput.m_AircraftMoveInfo.m_dSelfLon =1
    nextBlueInput.m_AircraftMoveInfo.m_fVN = 1
    nextRedInput.m_AircraftMoveInfo.m_fVN = 2
    nextRedInput.m_AircraftMoveInfo.m_dSelfAlt = 3
    nextBlueInput.m_AircraftMoveInfo.m_dSelfAlt = 4
    r = getReward(currentRedInput,
              currentBlueInput,
              currentRedOutput,
              currentBlueOutput,
              nextRedInput,
              nextBlueInput)
    axis = Axis()
    print(r)

    # import matplotlib.pyplot as plt
    # reward = []
    # for i in range(100, 9000, 100):
    #     redInput = AIPilotInput()
    #     redInput.m_AircraftMoveInfo.m_dSelfAlt = i
    #     blueInput = AIPilotInput()
    #     blueInput.m_AircraftMoveInfo.m_dSelfAlt = i-100
    #     reward.append(getAltitudeAdvantage(redInput, blueInput, 1))
    # plt.plot(reward)
    # plt.show()
    # print(reward)