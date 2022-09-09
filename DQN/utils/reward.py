from turtle import distance, speed
from utils.AI_Interface import *
from math import asin, acos, sin, cos, exp, pi
import numpy as np

class Axis():
    def __init__(self, a = 0, b = 0, c = 0):
        GxOy = np.array([[cos(a+pi/2) ,  -sin(a+pi/2),   0        ],
                         [sin(a+pi/2), cos(a+pi/2) ,   0        ],
                         [0           ,  0           ,   1       ]])
        GzOx = np.array([[cos(b)      ,  0           ,   -sin(b)  ],
                         [0           ,  1           ,   0       ],
                         [sin(b)     ,  0           ,   cos(b)  ]])
        GyOz = np.array([[1           ,  0           ,   0       ],
                         [0           ,  -cos(c)      ,   sin(c) ],
                         [0           ,  -sin(c)      ,   -cos(c)  ]])
        self.base = GxOy @ GzOx @ GyOz
    def transform(self, component, reverse = False):
        if reverse:
            return self.base @ component
        else:
            return self.base.T @ component

        

def getDistance(currentRedInput, currentBlueInput):
    distance = ((currentRedInput.m_AircraftMoveInfo.m_dSelfLon - currentBlueInput.m_AircraftMoveInfo.m_dSelfLon)**2 +\
                (currentRedInput.m_AircraftMoveInfo.m_dSelfLat - currentBlueInput.m_AircraftMoveInfo.m_dSelfLat)**2 +\
                (currentRedInput.m_AircraftMoveInfo.m_dSelfAlt - currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt)**2)**0.5
    return distance

def getSpeead(Input):
    speed = ((Input.m_AircraftMoveInfo.m_fVN)**2 +\
            (Input.m_AircraftMoveInfo.m_fVU)**2 +\
            (Input.m_AircraftMoveInfo.m_fVE)**2)**0.5
    return speed

def getAngleAdvantage(currentRedInput, currentBlueInput, versionID):
    distance = getDistance(currentRedInput, currentBlueInput)
    redSpeed = getSpeead(currentRedInput)
    blueSpeed = getSpeead(currentBlueInput)
    redAxis = Axis(currentRedInput.m_AircraftMoveInfo.m_fYaw,
                   currentRedInput.m_AircraftMoveInfo.m_fPitch,
                   currentRedInput.m_AircraftMoveInfo.m_fRoll)
    blueAxis = Axis(currentBlueInput.m_AircraftMoveInfo.m_fYaw,
                    currentBlueInput.m_AircraftMoveInfo.m_fPitch,
                    currentBlueInput.m_AircraftMoveInfo.m_fRoll)            
    if versionID == 0:
        positionVector = np.array([currentBlueInput.m_AircraftMoveInfo.m_dSelfLat -currentRedInput.m_AircraftMoveInfo.m_dSelfLat,
                                   currentBlueInput.m_AircraftMoveInfo.m_dSelfLon -currentRedInput.m_AircraftMoveInfo.m_dSelfLon,
                                   currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt -currentRedInput.m_AircraftMoveInfo.m_dSelfAlt])
        transformedPositionVectorRed = redAxis.transform(positionVector)
        redSightAngle = acos(transformedPositionVectorRed[0]/(distance+1e-4))
        transformedPositionVectorBlue = blueAxis.transform(-positionVector)
        blueSightAngle = acos(transformedPositionVectorBlue[0]/(distance+1e-4))
        reward = (blueSightAngle - redSightAngle)/(2*pi)+0.5
    elif versionID == 1 :
        phiRed = acos((currentRedInput.m_AircraftMoveInfo.m_fVN * (currentBlueInput.m_AircraftMoveInfo.m_dSelfLon - currentRedInput.m_AircraftMoveInfo.m_dSelfLon) +\
                       currentRedInput.m_AircraftMoveInfo.m_fVU * (currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt - currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) +\
                       currentRedInput.m_AircraftMoveInfo.m_fVE * (currentBlueInput.m_AircraftMoveInfo.m_dSelfLat - currentRedInput.m_AircraftMoveInfo.m_dSelfLat))/(redSpeed*distance+1e-4))
        phiBlue = acos((currentBlueInput.m_AircraftMoveInfo.m_fVN * (currentBlueInput.m_AircraftMoveInfo.m_dSelfLon - currentRedInput.m_AircraftMoveInfo.m_dSelfLon) +\
                        currentBlueInput.m_AircraftMoveInfo.m_fVU * (currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt - currentRedInput.m_AircraftMoveInfo.m_dSelfAlt) +\
                        currentBlueInput.m_AircraftMoveInfo.m_fVE * (currentBlueInput.m_AircraftMoveInfo.m_dSelfLat - currentRedInput.m_AircraftMoveInfo.m_dSelfLat))/(blueSpeed*distance+1e-4))
        reward = 1 - (abs(phiBlue)+abs(phiRed))/(2*acos(-1))
    return reward

def getDistanceAdvantage(currentRedInput, currentBlueInput, versionID):
    if versionID == 1:
        optimalDistance = 200
        D0 = 2000 #D0用于调整奖励梯度
        distance = getDistance(currentRedInput, currentBlueInput)
        reward = exp(-(distance - optimalDistance)/D0)
    return reward

def getAltitudeAdvantage(currentRedInput, currentBlueInput, versionID):
    if versionID == 1:
        optimalAlt = 6000
        if currentRedInput.m_AircraftMoveInfo.m_dSelfAlt < currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt:
            reward = currentRedInput.m_AircraftMoveInfo.m_dSelfAlt / (currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt+1e-4) - 0.5
        else:
            reward = exp(-abs(currentRedInput.m_AircraftMoveInfo.m_dSelfAlt - optimalAlt)/(currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt+1e-4)) #最佳空战高度6000
    return reward

def getSpeedAdvantage(currentRedInput, currentBlueInput, versionID):
    if versionID == 1:
        redSpeed = getSpeead(currentRedInput)
        blueSpeed = getSpeead(currentBlueInput)
        optimalSpeed = 200
        if 1.5*blueSpeed < optimalSpeed: #最佳空战速度200
            if redSpeed > optimalSpeed:
                reward = exp((optimalSpeed - redSpeed)/optimalSpeed)
            elif 1.5*blueSpeed < redSpeed:
                reward = 1
            elif 0.5*blueSpeed < redSpeed:
                reward = redSpeed/(blueSpeed+1e-4) -0.5
            else:
                reward = 0.5
        else:
            if redSpeed > optimalSpeed:
                reward = exp((optimalSpeed-redSpeed)/optimalSpeed)
            elif 0.5*blueSpeed < redSpeed and 1.5*blueSpeed >= redSpeed:
                reward = 0.4*(redSpeed/optimalSpeed -redSpeed/(blueSpeed+1e-4))
            elif 0.5*blueSpeed >= redSpeed:
                reward = 0.1
    return reward

def SRadarDetected(currentRedInput, currentBlueInput, currentRedOutput, currentBlueOutput):
    distance = getDistance(currentRedInput, currentBlueInput)

    if not currentRedInput.m_FCInfo.m_eAircraftMainState:  # 视距内
        speedPitch = asin((currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt -currentRedInput.m_AircraftMoveInfo.m_dSelfAlt)/(distance+1e-4))
        redAxis = Axis(currentRedInput.m_AircraftMoveInfo.m_fYaw,
                       speedPitch,
                       currentRedInput.m_AircraftMoveInfo.m_fRoll)
    else:
        redAxis = Axis(currentRedInput.m_AircraftMoveInfo.m_fYaw,
                       currentRedInput.m_AircraftMoveInfo.m_fPitch,
                       currentRedInput.m_AircraftMoveInfo.m_fRoll)
    positionVector = np.array([currentBlueInput.m_AircraftMoveInfo.m_dSelfLat -currentRedInput.m_AircraftMoveInfo.m_dSelfLat,
                               currentBlueInput.m_AircraftMoveInfo.m_dSelfLon -currentRedInput.m_AircraftMoveInfo.m_dSelfLon,
                               currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt -currentRedInput.m_AircraftMoveInfo.m_dSelfAlt])
    transformedPositionVectorRed = redAxis.transform(positionVector)

    positionEle = asin(transformedPositionVectorRed[2]/(distance+1e-4))
    positionAzi = acos(transformedPositionVectorRed[0]/(distance*cos(positionEle)+1e-4))

    if currentRedInput.m_FCInfo.m_eAircraftMainState:  # 视距内
        if positionAzi > -5 and positionAzi < 5 and \
            positionEle >= -7 and positionEle <= 53 :
            return True
        else:
            return False
    else:
        if positionAzi  > currentRedOutput.m_RadarCtrlCmd.m_fAziScanCent - currentRedOutput.m_RadarCtrlCmd.m_eAziScanRange/2 and\
            positionAzi < currentRedOutput.m_RadarCtrlCmd.m_fAziScanCent + currentRedOutput.m_RadarCtrlCmd.m_eAziScanRange/2 and\
            positionEle > currentRedOutput.m_RadarCtrlCmd.m_fEleScanCent - 4.5 * currentRedOutput.m_RadarCtrlCmd.m_eEleScanLine/2 and\
            positionEle < currentRedOutput.m_RadarCtrlCmd.m_fEleScanCent + 4.5 * currentRedOutput.m_RadarCtrlCmd.m_eEleScanLine/2 :
            return True
        else:
            return False

def SMRAAMCaptured(currentRedInput, currentBlueInput):
    for i in range(currentRedInput.m_MRAAMDataMsgSet.m_uiMsgNum):
        if currentRedInput.m_MRAAMDataMsgSet.m_MRAAMDataMsg[i].m_bCapture:
            return True
        else:
            return False

def SMRAAMCapturedInRange(currentRedInput, currentBlueInput):
    distance = getDistance(currentRedInput, currentBlueInput)
    if (SMRAAMCaptured(currentRedInput, currentBlueInput)):
        if (distance > currentRedInput.m_FCInfo.m_fRmin and \
            distance < currentRedInput.m_FCInfo.m_fRmax):
            if (distance < currentRedInput.m_FCInfo.m_fRnoescape):
                return 3
            else:
                return 2
        else:
            return 1
    else:
        return 0

def SRadarLocked(currentRedInput, currentBlueInput):
    maxRadarLockedTolerence = 10 #人为设定的雷达误差
    minRadarLockedDistance = 100000
    distance = getDistance(currentRedInput, currentBlueInput)
    if not currentRedInput.m_FCInfo.m_eAircraftMainState:  # 视距内
        speedPitch = asin((currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt -currentRedInput.m_AircraftMoveInfo.m_dSelfAlt)/(distance+1e-4))
        redAxis = Axis(currentRedInput.m_AircraftMoveInfo.m_fYaw,
                       speedPitch,
                       currentRedInput.m_AircraftMoveInfo.m_fRoll)
    else:
        redAxis = Axis(currentRedInput.m_AircraftMoveInfo.m_fYaw,
                       currentRedInput.m_AircraftMoveInfo.m_fPitch,
                       currentRedInput.m_AircraftMoveInfo.m_fRoll)
    positionVector = np.array([currentBlueInput.m_AircraftMoveInfo.m_dSelfLat -currentRedInput.m_AircraftMoveInfo.m_dSelfLat,
                               currentBlueInput.m_AircraftMoveInfo.m_dSelfLon -currentRedInput.m_AircraftMoveInfo.m_dSelfLon,
                               currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt -currentRedInput.m_AircraftMoveInfo.m_dSelfAlt])
    transformedPositionVectorRed = redAxis.transform(positionVector)

    for i in range(currentRedInput.m_RadarInfo.m_uiTgtNum):
        SRadarLockedPositionAlt = currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtDis * \
                                  sin(currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtEle)
        projectedDistance = currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtDis * \
                            cos(currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtEle)
        SRadarLockedPositionLon = projectedDistance * \
                                  cos(currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtAzi)
        SRadarLockedPositionLat = projectedDistance * \
                                  sin(currentRedInput.m_RadarInfo.m_TgtInfo[i].m_fTgtAzi)
        SRadarLockedDistance = (((SRadarLockedPositionLat - transformedPositionVectorRed[0])**2) + \
                                ((SRadarLockedPositionLon - transformedPositionVectorRed[1])**2) + \
                                ((SRadarLockedPositionAlt - transformedPositionVectorRed[2])**2))**0.5
        if SRadarLockedDistance < minRadarLockedDistance:
            minRadarLockedDistance = SRadarLockedDistance
    if minRadarLockedDistance < maxRadarLockedTolerence:
        return True
    else:
        return False

def getReward(previousRedInput,
              previousBlueInput,
              currentRedOutput,
              currentBlueOutput,
              currentRedInput,
              currentBlueInput):
    reward = 0
    if (previousRedInput.m_AircraftBasicInfo.m_bAlive and \
        (not currentRedInput.m_AircraftBasicInfo.m_bAlive)):
        return -1#被击毁
    if (previousRedInput.m_AircraftBasicInfo.m_fFuel > 0 and \
        currentRedInput.m_AircraftBasicInfo.m_fFuel <=0):
        return -1#油尽
    if (previousRedInput.m_AircraftMoveInfo.m_dSelfAlt > 0 and \
        currentRedInput.m_AircraftMoveInfo.m_dSelfAlt):
        return -1#撞地
    if (previousBlueInput.m_AircraftBasicInfo.m_bAlive and \
        (not currentBlueInput.m_AircraftBasicInfo.m_bAlive)):
        return 1#击毁敌机
    if (previousBlueInput.m_AircraftBasicInfo.m_fFuel > 0 and \
        currentBlueInput.m_AircraftBasicInfo.m_fFuel <=0):
        return 1#敌机油尽
    if (previousBlueInput.m_AircraftMoveInfo.m_dSelfAlt > 0 and \
        currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt):
        return 1#敌机撞地
    if (previousRedInput.m_AircraftBasicInfo.m_bAlive and \
        (not currentRedInput.m_AircraftBasicInfo.m_bAlive) and \
        previousBlueInput.m_AircraftBasicInfo.m_bAlive and \
        (not currentBlueInput.m_AircraftBasicInfo.m_bAlive)):
        return 0#同归于尽

    if (SRadarDetected(currentRedInput, currentBlueInput, currentRedOutput, currentBlueOutput)):
        reward += 0.05
    if (SRadarDetected(currentBlueInput, currentRedInput, currentBlueOutput, currentRedOutput)):
        reward -= 0.05

    if (SRadarLocked(currentRedInput, currentBlueInput)):
        reward += 0.2#火控雷达探测到信息
    if (SRadarLocked(currentBlueInput, currentRedInput)):
        reward -= 0.2#被火控雷达探测到信息

    capturedInRange = SMRAAMCapturedInRange(currentRedInput, currentBlueInput)
    if (capturedInRange == 1):#捕获但未进入射程
        reward += 0.3
    elif (capturedInRange == 2):#进入射程可逃逸
        reward += 0.55
    elif (capturedInRange == 3):#进入射程不可逃逸
        reward += 0.9

    capturedInRange = SMRAAMCapturedInRange(currentBlueInput, currentRedInput)
    if (capturedInRange == 1):#捕获但未进入射程
        reward -= 0.3
    elif (capturedInRange == 2):#进入射程可逃逸
        reward -= 0.55
    elif (capturedInRange == 3):#进入射程不可逃逸
        reward -= 0.9

    reward += 0.005 * getAngleAdvantage(currentRedInput, currentBlueInput, 1)
    reward += 0.008 * getSpeedAdvantage(currentRedInput, currentBlueInput, 1)
    reward += 0.0001 * getDistanceAdvantage(currentRedInput, currentBlueInput, 1)
    reward += 0.0001 * getDistanceAdvantage(currentRedInput, currentBlueInput, 1)

    return reward

if __name__ == '__main__':
    previousRedInput = AIPilotInput()
    previousBlueInput = AIPilotInput()
    currentRedOutput = AIPilotOutput()
    currentBlueOutput = AIPilotOutput()
    currentRedInput = AIPilotInput()
    currentBlueInput = AIPilotInput()

    currentBlueInput.m_AircraftMoveInfo.m_dSelfLon =1
    currentBlueInput.m_AircraftMoveInfo.m_fVN = 1
    currentRedInput.m_AircraftMoveInfo.m_fVN = 2
    currentRedInput.m_AircraftMoveInfo.m_dSelfAlt = 3
    currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt = 4
    getReward(previousRedInput,
              previousBlueInput,
              currentRedOutput,
              currentBlueOutput,
              currentRedInput,
              currentBlueInput)
    axis = Axis()
