from math import cos, pi
from utils.reward import *

MAX_TGT_NUM = 8
MAX_MRAAM_NUM = 6
MAX_SRAAM_NUM = 6
MAX_AAM_NUM_IN_SCENE = 12
Enum_RadarWorkMode_OFF = 0
Enum_RadarWorkMode_BVR = 1
Enum_RadarWorkMode_WVR = 2
Enum_AircraftTaskMode_BVR = 0
Enum_AircraftTaskMode_WVR = 1
Enum_ESMTgtType_Invalid = 0
Enum_ESMTgtType_Missile = 1
Enum_ESMTgtType_Plane	= 2
Enum_ESMTgtIff_Friend	= 0
Enum_ESMTgtIff_Unknown	= 1
Enum_ESMTgtIff_Foe		= 2
Enum_AircraftIff_Red	= 0
Enum_AircraftIff_Blue	= 1
Enum_AircraftIff_White	= 2
Enum_AAMType_MRAAM = 0
Enum_AAMType_SRAAM = 1
Enum_AAMState_Unfired = 0,
Enum_AAMState_Flying = 1
Enum_AAMState_Hit = 2
Enum_AAMState_Lost = 3
Enum_RadarOnOff_OFF = 0
Enum_RadarOnOff_ON	= 1
Enum_RadarEleScanLine_2	= 2
Enum_RadarEleScanLine_4	= 4
Enum_RadarAziScanRange_30	= 30
Enum_RadarAziScanRange_60	= 60
Enum_RadarAziScanRange_120	= 120
Earth_Radius = 6371004

class SAircraftBasicInfo_AI_Interface():
	def __init__(self):
		self.m_ID			= 0
		self.m_lTimeStamp	= 0
		self.m_bAlive		= True
		self.m_fFuel		= 0

class SAircraftMoveInfo_AI_Interface():
	def __init__(self):
		self.m_dSelfLon		= 0
		self.m_dSelfLat		= 0
		self.m_dSelfAlt		= 0
		self.m_fVN			= 0
		self.m_fVU			= 0
		self.m_fVE			= 0
		self.m_fAccN		= 0
		self.m_fAccU		= 0
		self.m_fAccE		= 0
		self.m_fAccBX		= 0
		self.m_fAccBY		= 0
		self.m_fAccBZ		= 0
		self.m_fTASpeed		= 0
		self.m_fMach		= 0
		self.m_fNormalAccel	= 0
		self.m_fYaw			= 0
		self.m_fPitch		= 0
		self.m_fRoll		= 0
		self.m_fAlpha		= 0
		self.m_fBeta		= 0
		self.m_fCrab		= 0
		self.m_fOmegaYaw	= 0
		self.m_fOmegaPitch	= 0
		self.m_fOmegaRoll	= 0

class SRadarTgtInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtLot	= 0
		self.m_fTgtDis	= 0
		self.m_fTgtAzi	= 0
		self.m_fTgtEle	= 0
		self.m_fTgtVN	= 0
		self.m_fTgtVU	= 0
		self.m_fTgtVE	= 0
		self.m_fTgtAccN	= 0
		self.m_fTgtAccU	= 0
		self.m_fTgtAccE	= 0
		self.m_fTgtDisDot= 0

class SRadarInfo_AI_Interface():
	def __init__(self):
		self.m_eRadarWorkMode = Enum_RadarWorkMode_OFF
		self.m_fAziScanRange = 0
		self.m_fEleScanRange = 0
		self.m_fAziScanCent = 0
		self.m_fEleScanCent = 0
		self.m_uiTgtNum = 0
		self.m_TgtInfo = []
		for i in range(MAX_TGT_NUM):
			self.m_TgtInfo.append(SRadarTgtInfo_AI_Interface())

class SHMDTgtInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtLot	= 0
		self.m_fTgtDis	= 0
		self.m_fTgtAziC	= 0
		self.m_fTgtEleC	= 0
		self.m_fYaw		= 0
		self.m_fPitch	= 0
		self.m_fRoll	= 0
		self.m_fTgtVN	= 0
		self.m_fTgtVU	= 0
		self.m_fTgtVE	= 0
		self.m_fTgtAccN	= 0
		self.m_fTgtAccU	= 0
		self.m_fTgtAccE	= 0
		self.m_fTgtDisDot= 0

class SHMDInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtNum = 0
		self.m_TgtInfo = []
		for i in range(MAX_TGT_NUM):
			self.m_TgtInfo.append(SHMDTgtInfo_AI_Interface())

class SFCInfo_AI_Interface():
	def __init__(self):
		self.m_eAircraftMainState = Enum_AircraftTaskMode_BVR
		self.m_fRmax		= 0.0
		self.m_fRnoescape	= 0.0
		self.m_fRmin		= 0.0
		self.m_bINRNG		= False
		self.m_bSHOOT		= False
		self.m_bWeaponReady	= False

class SESMTgtInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtLot	= 0
		self.m_eTgtType	= Enum_ESMTgtType_Invalid
		self.m_eTgtIff	= Enum_ESMTgtIff_Unknown
		self.m_fTgtAzi	= 0.0
		self.m_fTgtEle	= 0.0

class SESMInfo_AI_Interface():
	def __init__(self):
		self.m_uiAlarmTgtNum = 0
		self.m_AlarmTgtInfo = []
		for i in range(MAX_TGT_NUM):
			self.m_AlarmTgtInfo.append(SESMTgtInfo_AI_Interface())

class SDASTgtInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtLot	= 0
		self.m_fTgtAzi	= 0.0
		self.m_fTgtEle	= 0.0

class SDASInfo_AI_Interface():
	def __init__(self):
		self.m_uiThreatTgtNum = 0
		self.m_ThreatTgtInfo = []
		for i in range(MAX_TGT_NUM):
			self.m_ThreatTgtInfo.append(SDASTgtInfo_AI_Interface())

class SAWACSTgtInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtLot	= 0
		self.m_eIFF		= Enum_AircraftIff_White
		self.m_fTgtLon	= 0
		self.m_fTgtLat	= 0
		self.m_fTgtAlt	= 0
		self.m_fTgtVN	= 0
		self.m_fTgtVU	= 0
		self.m_fTgtVE	= 0

class SAWACSInfo_AI_Interface():
	def __init__(self):
		self.m_uiTgtNum = 0
		self.m_TgtInfo = []
		for i in range(MAX_TGT_NUM):
			self.m_TgtInfo.append(SAWACSTgtInfo_AI_Interface())

class SMRAAMDataMsg_AI_Interface():
	def __init__(self):
		self.m_uiAAMID		= 0
		self.m_bSeekerOpen	= False
		self.m_bCapture		= False
		self.m_dLon			= 0
		self.m_dLat			= 0
		self.m_dAlt			= 0
		self.m_fMslVX		= 0
		self.m_fMslVU		= 0
		self.m_fMslVE		= 0

class SMRAAMDataMsgSet_AI_Interface():
	def __init__(self):
		self.m_uiMsgNum = 0
		self.m_MRAAMDataMsg = []
		for i in range(MAX_TGT_NUM):
			self.m_MRAAMDataMsg.append(SMRAAMDataMsg_AI_Interface())

class SAAMData_AI_Interface():
	def __init__(self):
		self.m_uiAAMID		= 0
		self.m_eAAMType		= Enum_AAMType_MRAAM
		self.m_uiPlaneID	= 0
		self.m_eAAMState	= Enum_AAMState_Unfired
		self.m_bSeekerOpen	= False
		self.m_bCapture		= False
		self.m_dLon			= 0
		self.m_dLat			= 0
		self.m_dAlt			= 0
		self.m_fMslVX		= 0
		self.m_fMslVU		= 0
		self.m_fMslVE		= 0
		self.m_fMslYaw		= 0
		self.m_fMslPitch	= 0
		self.m_fTgtDis		= 0

class SAAMDataSet_AI_Interface():
	def __init__(self):
		self.m_iAAMNum = 0
		self.m_AAMData = []
		for i in range(MAX_AAM_NUM_IN_SCENE):
			self.m_AAMData.append(SAAMData_AI_Interface())

class AIPilotInput():
	def __init__(self):
		self.m_AircraftBasicInfo = SAircraftBasicInfo_AI_Interface()
		self.m_AircraftMoveInfo =SAircraftMoveInfo_AI_Interface()
		self.m_RadarInfo =SRadarInfo_AI_Interface()
		self.m_HMDInfo =SHMDInfo_AI_Interface()
		self.m_FCInfo =SFCInfo_AI_Interface()
		self.m_ESMInfo =SESMInfo_AI_Interface()
		self.m_DASInfo =SDASInfo_AI_Interface()
		self.m_AWACSInfo =SAWACSInfo_AI_Interface()
		self.m_MRAAMDataMsgSet =SMRAAMDataMsgSet_AI_Interface()
		self.m_AAMDataSet =SAAMDataSet_AI_Interface()
		self.m_ManMadeFeature = ManMade_Feature()

class SFlyCtrlCmd_AI_Interface():
	def __init__(self):
		self.m_fStickLat	= 0.0
		self.m_fStickLon	= 0.0
		self.m_fThrottle	= 0.0
		self.m_fRudder		= 0.0

class SFCCtrlCmd_AI_Interface():
	def __init__(self):
		self.m_eMainTaskMode	= Enum_AircraftTaskMode_BVR

class SRadarCtrlCmd_AI_Interface():
	def __init__(self):
		self.m_eRadarOnOff = Enum_RadarOnOff_OFF
		self.m_eEleScanLine	= Enum_RadarEleScanLine_2
		self.m_eAziScanRange = Enum_RadarAziScanRange_30
		self.m_fAziScanCent	= 0
		self.m_fEleScanCent	= 0

class SWeaponCtrlCmd_AI_Interface():
	def __init__(self):
		self.m_bWeaponLaunch = False

class ManMade_Feature():
	def __init__(self):
		self.speed = 0
		self.phiRed =0

class AIPilotOutput():
	def __init__(self):
		self.m_FlyCtrlCmd = SFlyCtrlCmd_AI_Interface()
		self.m_FCCtrlCmd = SFCCtrlCmd_AI_Interface()
		self.m_RadarCtrlCmd = SRadarCtrlCmd_AI_Interface()
		self.m_WeaponCtrlCmd = SWeaponCtrlCmd_AI_Interface()

class AIPilotInit():
	def __init__(self):
		self.m_ID			= 0
		self.m_eIFF			= Enum_AircraftIff_Red
		self.m_dInitLon		= 0
		self.m_dInitLat		= 0
		self.m_dInitAlt		= 0
		self.m_fInitVelocity= 0.0
		self.m_fInitYaw		= 0.0
		self.m_fInitPitch	= 0.0
		self.m_fInitRoll	= 0.0
		self.m_uiMRAAMNum	= 0
		self.m_uiSRAAMNum	= 0
		self.m_fInitFuel	= 0

def getStateAndAction(row):
	input = AIPilotInput()
	output = AIPilotOutput()
	it = iter(row)

	#[0:4]
	input.m_AircraftBasicInfo.m_ID = next(it)
	input.m_AircraftBasicInfo.m_lTimeStamp = next(it)*6000
	input.m_AircraftBasicInfo.m_bAlive = next(it)
	input.m_AircraftBasicInfo.m_fFuel = next(it)*6000

	#[4:28]
	input.m_AircraftMoveInfo.m_dSelfLon = next(it)*128
	input.m_AircraftMoveInfo.m_dSelfLat = next(it)*32
	input.m_AircraftMoveInfo.m_dSelfAlt = next(it)*10000
	# print(input.m_AircraftMoveInfo.m_dSelfLon, input.m_AircraftMoveInfo.m_dSelfLat, input.m_AircraftMoveInfo.m_dSelfAlt)
	input.m_AircraftMoveInfo.m_dSelfLon,\
	input.m_AircraftMoveInfo.m_dSelfLat = (Earth_Radius+input.m_AircraftMoveInfo.m_dSelfAlt)*pi/180 * cos(input.m_AircraftMoveInfo.m_dSelfLat*pi/180) * input.m_AircraftMoveInfo.m_dSelfLon,\
										  (Earth_Radius+input.m_AircraftMoveInfo.m_dSelfAlt)*pi/180 * input.m_AircraftMoveInfo.m_dSelfLat
	input.m_AircraftMoveInfo.m_fVN = next(it)*700
	input.m_AircraftMoveInfo.m_fVU = next(it)*700
	input.m_AircraftMoveInfo.m_fVE = next(it)*700
	input.m_AircraftMoveInfo.m_fAccN = next(it)*100
	input.m_AircraftMoveInfo.m_fAccU = next(it)*100
	input.m_AircraftMoveInfo.m_fAccE = next(it)*100
	input.m_AircraftMoveInfo.m_fAccBX = next(it)*700
	input.m_AircraftMoveInfo.m_fAccBY = next(it)*700
	input.m_AircraftMoveInfo.m_fAccBZ = next(it)*700
	input.m_AircraftMoveInfo.m_fTASpeed = next(it)*700
	input.m_AircraftMoveInfo.m_fMach = next(it)*2.2
	input.m_AircraftMoveInfo.m_fNormalAccel = next(it)*10.0
	input.m_AircraftMoveInfo.m_fYaw = next(it)*4.0
	input.m_AircraftMoveInfo.m_fPitch = next(it)*4.0
	input.m_AircraftMoveInfo.m_fRoll = next(it)*4.0
	input.m_AircraftMoveInfo.m_fAlpha = next(it)*4.0
	input.m_AircraftMoveInfo.m_fBeta = next(it)*4.0
	input.m_AircraftMoveInfo.m_fCrab = next(it)*4.0
	input.m_AircraftMoveInfo.m_fOmegaYaw = next(it)*15.0
	input.m_AircraftMoveInfo.m_fOmegaPitch = next(it)*15.0
	input.m_AircraftMoveInfo.m_fOmegaRoll = next(it)*15.0

	#[28:33]
	input.m_RadarInfo.m_eRadarWorkMode = next(it) - 1
	input.m_RadarInfo.m_fAziScanRange = (next(it)+1)*30
	input.m_RadarInfo.m_fEleScanRange = next(it)
	input.m_RadarInfo.m_fAziScanCent = next(it)
	input.m_RadarInfo.m_fEleScanCent = next(it)

	#[33:122]
	input.m_RadarInfo.m_uiTgtNum = next(it)*8
	for radar_num in range(MAX_TGT_NUM):
		input.m_RadarInfo.m_TgtInfo[radar_num].m_uiTgtLot = next(it)*8
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtDis = next(it)*10000
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAzi = next(it)*4
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtEle = next(it)*4
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVN = next(it)*700
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVU = next(it)*700
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVE = next(it)*700
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccN = next(it)*100
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccU = next(it)*100
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccE = next(it)*100
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtDisDot = next(it)*700

	#[122:235]
	input.m_HMDInfo.m_uiTgtNum = next(it)*8
	for hmd_num in range(MAX_TGT_NUM):
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_uiTgtLot = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtDis = next(it)*10000
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAziC = next(it)*4
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtEleC = next(it)*4
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fYaw = next(it)*4
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fPitch = next(it)*4
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fRoll = next(it)*4
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVN = next(it)*700
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVU = next(it)*700
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVE = next(it)*700
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccN = next(it) * 100
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccU = next(it)*100
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccE = next(it)*100
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtDisDot = next(it)*700

	#[235:242]
	input.m_FCInfo.m_eAircraftMainState = next(it)
	input.m_FCInfo.m_fRmax = next(it)*30000
	input.m_FCInfo.m_fRnoescape = next(it)*30000
	input.m_FCInfo.m_fRmin = next(it)*6000
	input.m_FCInfo.m_bINRNG = next(it)
	input.m_FCInfo.m_bSHOOT = next(it)
	input.m_FCInfo.m_bWeaponReady = next(it)

	#[242:283]
	input.m_ESMInfo.m_uiAlarmTgtNum = next(it)*8
	for esm_num in range(MAX_TGT_NUM):
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_uiTgtLot = next(it)
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_eTgtType = next(it)+1
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_eTgtIff = next(it)+1
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_fTgtAzi = next(it)*1800
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_fTgtEle = next(it)*1800

	#[283:308]
	input.m_DASInfo.m_uiThreatTgtNum = next(it)*8
	for das_num in range(MAX_TGT_NUM):
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_uiTgtLot = next(it)
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_fTgtAzi = next(it)*1800
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_fTgtEle = next(it)*1800

	#[308:373]
	input.m_AWACSInfo.m_uiTgtNum = next(it)*8
	for awacs_num in range(MAX_TGT_NUM):
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_uiTgtLot = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_eIFF = next(it)+1
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLon = next(it)*128
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLat = next(it)*32
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtAlt = next(it)*11000
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLon ,\
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLat = (Earth_Radius+input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtAlt)*pi/180 * cos(input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLat*pi/180) * input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLon,\
										  	  			   (Earth_Radius+input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtAlt)*pi/180 * input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLat
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVN = next(it)*700
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVU = next(it)*700
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVE = next(it)*700

	#[373:428]
	input.m_MRAAMDataMsgSet.m_uiMsgNum = next(it)*16
	for mraamd_num in range(MAX_MRAAM_NUM):
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_uiAAMID = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_bSeekerOpen = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_bCapture = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLon = next(it)*128
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLat = next(it)*32
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dAlt = next(it)*10000
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLon ,\
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLat = (Earth_Radius+input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dAlt)*pi/180 * cos(input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLat*pi/180) * input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLon,\
										  	  			   			(Earth_Radius+input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dAlt)*pi/180 * input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLat
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVX = next(it)*500
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVU = next(it)*1000
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVE = next(it)*1000

	input.m_ManMadeFeature.speed = next(it) * 500
	input.m_ManMadeFeature.phiRed = 1 - next(it)
	#[430:611]
	input.m_AAMDataSet.m_iAAMNum = next(it)*12
	for aamd_num in range(MAX_AAM_NUM_IN_SCENE):
		input.m_AAMDataSet.m_AAMData[aamd_num].m_uiAAMID = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_eAAMType = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_uiPlaneID = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_eAAMState = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_bSeekerOpen = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_bCapture = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLon = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLat = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dAlt = next(it)*10000
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLon ,\
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLat = (Earth_Radius+input.m_AAMDataSet.m_AAMData[aamd_num].m_dAlt)*pi/180 * cos(input.m_AAMDataSet.m_AAMData[aamd_num].m_dLat*pi/180) * input.m_AAMDataSet.m_AAMData[aamd_num].m_dLon,\
										  	  			(Earth_Radius+input.m_AAMDataSet.m_AAMData[aamd_num].m_dAlt)*pi/180 * input.m_AAMDataSet.m_AAMData[aamd_num].m_dLat
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVX = next(it)*1000
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVU = next(it)*1000
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVE = next(it)*1000
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslYaw = next(it)*4
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslPitch = next(it)*4
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fTgtDis = next(it)*10000

	#[611:622]
	output.m_FlyCtrlCmd.m_fStickLat = next(it)
	output.m_FlyCtrlCmd.m_fStickLon = next(it)
	output.m_FlyCtrlCmd.m_fThrottle = next(it)
	output.m_FlyCtrlCmd.m_fRudder = next(it)
	output.m_FCCtrlCmd.m_eMainTaskMode = next(it)
	# output.m_FCCtrlCmd.m_eRadarOnOff = next(it)
	output.m_RadarCtrlCmd.m_eEleScanLine = next(it)
	output.m_RadarCtrlCmd.m_eAziScanRange = next(it)
	# output.m_RadarCtrlCmd.m_fAziScanCent = next(it)
	# output.m_RadarCtrlCmd.m_fEleScanCent = next(it)
	output.m_WeaponCtrlCmd.m_bWeaponLaunch = next(it)
	return input, output

def buildData(path):
	import pandas as pd
	import numpy as np
	data = pd.read_csv(path, header = None)
	n = data.shape[0]

	# redPlaneData = []
	# bluePlaneData = []

	transition = []
	previousRedInput, previousRedOutput = getStateAndAction(data.loc[0])
	previousBlueInput, previousBlueOutput = getStateAndAction(data.loc[1])
	redID = previousRedInput.m_AircraftBasicInfo.m_ID
	blueID = previousBlueInput.m_AircraftBasicInfo.m_ID
	
	# redPlaneData.append([previousRedInput.m_AircraftMoveInfo.m_dSelfLon,
	# 					previousRedInput.m_AircraftMoveInfo.m_dSelfLat,
	# 					previousRedInput.m_AircraftMoveInfo.m_dSelfAlt,
	# 					previousRedInput.m_AircraftMoveInfo.m_fYaw,
	# 					previousRedInput.m_AircraftMoveInfo.m_fPitch,
	# 					previousRedInput.m_AircraftMoveInfo.m_fRoll,])
	# bluePlaneData.append([previousBlueInput.m_AircraftMoveInfo.m_dSelfLon,
	# 					previousBlueInput.m_AircraftMoveInfo.m_dSelfLat,
	# 					previousBlueInput.m_AircraftMoveInfo.m_dSelfAlt,
	# 					previousBlueInput.m_AircraftMoveInfo.m_fYaw,
	# 					previousBlueInput.m_AircraftMoveInfo.m_fPitch,
	# 					previousBlueInput.m_AircraftMoveInfo.m_fRoll,])
	for i in range(2, n//2*2, 2):
		currentRedInput, currentRedOutput = getStateAndAction(data.loc[i])
		currentBlueInput, currentBlueOutput = getStateAndAction(data.loc[i+1])
		if currentRedInput.m_AircraftBasicInfo.m_ID == redID and\
		   currentBlueInput.m_AircraftBasicInfo.m_ID == blueID:
			transition.append([previousRedInput,
							   previousBlueInput,
							   previousRedOutput,
							   previousBlueOutput,
							   currentRedInput,
							   currentBlueInput])
			# redPlaneData.append([currentRedInput.m_AircraftMoveInfo.m_dSelfLon,
			# 					currentRedInput.m_AircraftMoveInfo.m_dSelfLat,
			# 					currentRedInput.m_AircraftMoveInfo.m_dSelfAlt,
			# 					currentRedInput.m_AircraftMoveInfo.m_fYaw,
			# 					currentRedInput.m_AircraftMoveInfo.m_fPitch,
			# 					currentRedInput.m_AircraftMoveInfo.m_fRoll,])
			# bluePlaneData.append([currentBlueInput.m_AircraftMoveInfo.m_dSelfLon,
			# 					currentBlueInput.m_AircraftMoveInfo.m_dSelfLat,
			# 					currentBlueInput.m_AircraftMoveInfo.m_dSelfAlt,
			# 					currentBlueInput.m_AircraftMoveInfo.m_fYaw,
			# 					currentBlueInput.m_AircraftMoveInfo.m_fPitch,
			# 					currentBlueInput.m_AircraftMoveInfo.m_fRoll,])
			previousRedInput = currentRedInput
			previousBlueInput = currentBlueInput
			previousRedOutput = currentRedOutput
			previousBlueOutput = currentBlueOutput
		else:
			break
	# redPlaneData = np.array(redPlaneData)
	# bluePlaneData = np.array(bluePlaneData)
	# return transition, redPlaneData.T, bluePlaneData.T
	return transition

def dataForTacview(path):
	import pandas as pd
	import numpy as np
	data = pd.read_csv(path, header = None)

	#F-14A
	dataRed = data.loc[data.iloc[:,0]==data.iloc[0,0],[1,4,5,6,21,20,19]]
	# dataRed = data.iloc[0::2,[1,4,5,6,21,20,19]]
	dataRed.iloc[:,0] = dataRed.iloc[:,0]/10000
	dataRed.iloc[:,1] = dataRed.iloc[:,1]*128
	dataRed.iloc[:,2] = dataRed.iloc[:,2]*32
	dataRed.iloc[:,3] = dataRed.iloc[:,3]*10000
	dataRed.iloc[:,4] = dataRed.iloc[:,4]*4*180/pi
	dataRed.iloc[:,5] = dataRed.iloc[:,5]*4*180/pi
	dataRed.iloc[:,6] = -dataRed.iloc[:,6]*4*180/pi
	dataRed.columns = ['Time','Longitude','Latitude','Altitude','Roll (deg)','Pitch (deg)','Yaw (deg)']
	dataRed.to_csv('F-16A [Red].csv', index = None)

	dataBlue = data.loc[data.iloc[:,0]==data.iloc[1,0],[1,4,5,6,21,20,19]]
	# dataBlue = data.iloc[1::2,[1,4,5,6,21,20,19]]
	dataBlue.iloc[:,0] = dataBlue.iloc[:,0]/10000
	dataBlue.iloc[:,1] = dataBlue.iloc[:,1]*128
	dataBlue.iloc[:,2] = dataBlue.iloc[:,2]*32
	dataBlue.iloc[:,3] = dataBlue.iloc[:,3]*10000
	dataBlue.iloc[:,4] = dataBlue.iloc[:,4]*4*180/pi
	dataBlue.iloc[:,5] = dataBlue.iloc[:,5]*4*180/pi
	dataBlue.iloc[:,6] = -dataBlue.iloc[:,6]*4*180/pi
	dataBlue.columns = ['Time','Longitude','Latitude','Altitude','Roll (deg)','Pitch (deg)','Yaw (deg)']
	dataBlue.to_csv('F-16A [Blue].csv', index = None)

	# AIM-9X
	for i in range(MAX_AAM_NUM_IN_SCENE):
		missle_i = data.iloc[:,[1, 431+i*15+6, 431+i*15+7, 431+i*15+8, 431+i*15+13, 431+i*15+12]].copy()
		missle_i.iloc[:,0] = missle_i.iloc[:,0]/10000
		missle_i.iloc[:,1] = missle_i.iloc[:,1]
		missle_i.iloc[:,2] = missle_i.iloc[:,2]
		missle_i.iloc[:,3] = missle_i.iloc[:,3]*10000
		missle_i.iloc[:,4] = missle_i.iloc[:,4]*4*180/pi
		missle_i.iloc[:,5] = -missle_i.iloc[:,5]*4*180/pi
		missle_i.columns = ['Time','Longitude','Latitude','Altitude','Pitch (deg)','Yaw (deg)']
		# missle_i.to_csv('R-27ET [{}].csv'.format(['Red', 'Blue'][]), index = None)
		missle_i.to_csv('AIM-9X ({}) [{}].csv'.format(i, 'Black'), index = None)

def plotReward(transition):
	IncidentReward = [[],[]]
	AngleReward = [[],[]]
	DistanceReward = [[],[]]
	AAMHitReward = [[],[]]
	StabilityReward = [[],[]]
	AAMCapturedInRange = [[],[]]
	SpeedReward = [[],[]]
	RadarDetectRangeReward = [[],[]]
	RadarLockedReward = [[],[]]
	AltReward = [[],[]]
	TotalReward = [[],[]]
	labels = ['angle','distance','AAM','Stability','Incident','AAMCaptured', 'Speed', 'RadarDetectRange', 'RadarLocked', 'Alt']
	fig = plt.figure()
	for planeId in range(2):
		for i in transition:
			Angle_i = getAngleAdvantage(i[planeId%2], i[(planeId+1)%2], 1)*0.1
			Distance_i = getDistanceAdvantage(i[planeId%2], i[(planeId+1)%2], 2)*0.001
			AAMHit_i = AAMHit(i[planeId])*0.1
			Stability_i = getStabilityReward(i[planeId+4], i[planeId+2])*0.1
			Incident_i = getIncidentReward(i[planeId%2], i[(planeId+1)%2], i[planeId%2+4], i[(planeId+1)%2+4])
			AAMCapturedInRange_i = SMRAAMCapturedInRange(i[planeId], i[(planeId+1)%2])
			Speed_i = getSpeedAdvantage(i[planeId%2+4], i[(planeId+1)%2+4], 2)*0.05
			RadarDetect_i = SRadarDetectRange(i[planeId%2+4], i[(planeId+1)%2+4], i[planeId%2+2], i[(planeId+1)%2+2])*0.02
			RadarLocked_i = SRadarLocked(i[planeId%2+4], i[(planeId+1)%2+4])*0.05
			Alt_i = getAltitudeAdvantage(i[planeId%2+4], i[(planeId+1)%2+4], i[planeId%2], 2)*0.15

			AngleReward[planeId].append(Angle_i)
			DistanceReward[planeId].append(Distance_i)
			AAMHitReward[planeId].append(AAMHit_i)
			StabilityReward[planeId].append(Stability_i)
			IncidentReward[planeId].append(Incident_i)
			AAMCapturedInRange[planeId].append(AAMCapturedInRange_i)
			SpeedReward[planeId].append(Speed_i)
			RadarDetectRangeReward[planeId].append(RadarDetect_i)
			RadarLockedReward[planeId].append(RadarLocked_i)
			AltReward[planeId].append(Alt_i)
			TotalReward[planeId].append(Angle_i + \
										Distance_i + \
										AAMHit_i + \
										Stability_i + \
										Incident_i + \
										AAMCapturedInRange_i + \
										Speed_i + \
										RadarDetect_i + \
										RadarLocked_i + \
										Alt_i)

		ax = fig.add_subplot(1,2,planeId+1)
		ax.set_title('{} Reward'.format(['red','blue'][planeId]))
		ax.plot(AngleReward[planeId], label = labels[0])
		ax.plot(DistanceReward[planeId], label = labels[1])
		ax.plot(AAMHitReward[planeId], label = labels[2])
		ax.plot(StabilityReward[planeId], label = labels[3])
		ax.plot(IncidentReward[planeId], label = labels[4])
		ax.plot(AAMCapturedInRange[planeId], label = labels[5])
		ax.plot(SpeedReward[planeId], label = labels[6])
		ax.plot(RadarDetectRangeReward[planeId], label = labels[7])
		ax.plot(RadarLockedReward[planeId], label = labels[8])
		ax.plot(AltReward[planeId], label = labels[9])
		ax.plot(TotalReward[planeId], label = 'total')
		ax.legend()
	plt.show()

	# 	Angle_i = getAngleAdvantage(i[5], i[4], 1)*0.05
	# 	Distance_i = getDistanceAdvantage(i[5], i[4], 2)*0.05
	# 	AngleReward.append(Angle_i)
	# 	DistanceReward.append(Distance_i)
	# 	TotalReward.append(Angle_i + Distance_i)

	# fig = plt.figure()
	# ax1 = fig.add_subplot(1,2,1)
	# ax1.set_title('Red Reward')
	# ax1.plot(AngleReward, label = 'angle')
	# ax1.plot(DistanceReward, label = 'distance')
	# ax1.plot(TotalReward, label = 'total')
	# ax1.legend()
	# ax2 = fig.add_subplot(1,2,2)
	# ax2.set_title('Blue Reward')
	# ax2.plot(AngleReward, label = 'angle')
	# ax2.plot(DistanceReward, label = 'distance')
	# ax2.plot(TotalReward, label = 'total')
	# ax2.legend()
	# plt.savefig('reward.jpg')
	# plt.show()


if __name__ == '__main__':
	# AIPilotInit()
	# AIPilotOutput()
	# AIPilotOutput()

# m_RadarInfo.m_uiTgtNum
# m_HMDInfo.m_uiTgtNum
# m_ESMInfo.m_uiAlarmTgtNum
# m_DASInfo.m_uiThreatTgtNum
# m_AWACSInfo.m_uiTgtNum
# m_MRAAMDataMsgSet.m_uiMsgNum
# m_AAMDataSet.m_iAAMNum

	import pandas as pd
	import numpy as np
	from matplotlib.animation import FuncAnimation
	# from visual import Visualization
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	# from reward import *
	import pickle as pkl
	

	# data = pd.read_csv('testdata', header = None).iloc[0,:]
	# redInput0, redOuput0 = getStateAndAction(data)
	# print(redInput0.m_AircraftMoveInfo.m_dSelfLon, redInput0.m_AircraftMoveInfo.m_dSelfLat, redInput0.m_AircraftMoveInfo.m_dSelfAlt)
	# data = pd.read_csv('testdata', header = None).iloc[2,:]
	# redInput1, redOutput1 = getStateAndAction(data)
	# print(redInput1.m_AircraftMoveInfo.m_dSelfLon, redInput1.m_AircraftMoveInfo.m_dSelfLat, redInput1.m_AircraftMoveInfo.m_dSelfAlt)
	# print(getDistance(redInput0, redInput1), getSpeead(redInput0)*0.02)

	dataForTacview('data8')

	# transition = buildData('data')
	# with open('transition.pkl', 'wb') as f:
	# 	pkl.dump(transition, f)

	# with open('transition.pkl', 'rb') as f:
	# 	transition = pkl.load(f)
	
	# plotReward(transition)

	# for i in transition:
	# 	if i[2].m_FlyCtrlCmd.m_fThrottle < 0 and i[2].m_FlyCtrlCmd.m_fThrottle >1:
	# 		print(i[2].m_FlyCtrlCmd.m_fThrottle)
	
	# testRedInput = transition[0][0]
	# testBlueInput = transition[0][1]
	# print(testRedInput.m_FCInfo.m_fRmax)

	# j = 7
	# for i in range(len(transition)):
	# 	# for j in range(6, int(transition[i][5].m_AAMDataSet.m_iAAMNum)):
	# 	dist = ((transition[i][5].m_AAMDataSet.m_AAMData[j].m_dLon - transition[i][4].m_AircraftMoveInfo.m_dSelfLon)**2+\
	# 			(transition[i][5].m_AAMDataSet.m_AAMData[j].m_dLat - transition[i][4].m_AircraftMoveInfo.m_dSelfLat)**2+\
	# 			(transition[i][5].m_AAMDataSet.m_AAMData[j].m_dAlt - transition[i][4].m_AircraftMoveInfo.m_dSelfAlt)**2)**0.5
	# 	print(dist, transition[i][5].m_AAMDataSet.m_AAMData[j].m_fTgtDis)

	# dist = []
	# travel_dist = []
	# for i in range(len(transition)):
	# 	dist.append(getDistance(transition[i][0], transition[i][4]))
	# 	travel_dist.append(getSpeead(transition[i][0])*0.01)
		# dist = getDistance(transition[i][0], transition[i][4])
		# travel_dist = getSpeead(transition[i][0])*0.01
		# print(travel_dist, dist)
	# plt.figure(figsize=(50,10))
	# plt.plot(dist, label = 'dist')
	# plt.plot(travel_dist, label = 'travel dist')
	# plt.legend()
	# plt.show()
	
	# 	if transition[i][4].m_FCInfo.m_fRmax>0 or transition[i][5].m_FCInfo.m_fRmax>0:
	# 		print(transition[i][4].m_FCInfo.m_fRmax, transition[i][5].m_FCInfo.m_fRmax, i)
			# print(transition[i+1][0].m_FCInfo.m_fRmin, transition[i+1][1].m_FCInfo.m_fRmin)
			# print()
			# for j in range(transition[i][4].m_HMDInfo.m_uiTgtNum):
			# 	print(getDistance(transition[i][4], transition[i][5]))
			# 	print(transition[i][4].m_HMDInfo.m_TgtInfo[j].m_fTgtDis, end = ' ')
			# 	print()

	# angleReward = []
	# distanceReward = []
	# AAMHitReward = []
	# StabilityReward = []
	# totalReward = []

	# distance = []
	# x_r = []
	# y_r = []
	# z_r = []
	# x_b = []
	# y_b = []
	# z_b = []

	# for i in transition:
	# 	angle_i = getAngleAdvantage(i[5], i[4], 1)*0.05
	# 	distance_i = getDistanceAdvantage(i[5], i[4], 2)*0.001
	# 	AAMHit_i = AAMHit(i[5])*0.1
	# 	stability_i = getStabilityReward(i[5], i[3])*0.05
	# 	angleReward.append(angle_i)
	# 	distanceReward.append(distance_i)
	# 	AAMHitReward.append(AAMHit_i)
	# 	StabilityReward.append(stability_i)
	# 	totalReward.append(angle_i + distance_i + AAMHit_i + stability_i)

		# distance.append(getDistance(i[4], i[5]))

		# x_r.append(i[4].m_AircraftMoveInfo.m_dSelfLon)
		# y_r.append(i[4].m_AircraftMoveInfo.m_dSelfLat)
		# z_r.append(i[4].m_AircraftMoveInfo.m_dSelfAlt)
		# x_b.append(i[5].m_AircraftMoveInfo.m_dSelfLon)
		# y_b.append(i[5].m_AircraftMoveInfo.m_dSelfLat)
		# z_b.append(i[5].m_AircraftMoveInfo.m_dSelfAlt)

	# plt.plot(angleReward, label = 'angle')
	# plt.plot(distanceReward, label = 'distance')
	# plt.plot(totalReward, label = 'total')
	# plt.plot(AAMHitReward, label = 'AAM')
	# plt.plot(StabilityReward, label = 'stability')
	# plt.legend()

	# # plt.plot(distance)

	# plt.savefig('reward.jpg')
	# plt.show()

	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot(x_r, y_r, z_r)
	# ax.plot(x_b, y_b, z_b)
	# plt.show()

	# timeSpan = redPlaneData.shape[1]
	# maxSimRange = np.array([redPlaneData.max(1)[:3], bluePlaneData.max(1)[:3]]).max(0)
	# minSimRange = np.array([redPlaneData.min(1)[:3], bluePlaneData.min(1)[:3]]).min(0)
	# simRange = np.array([minSimRange, maxSimRange]).T

	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.set_xlim(simRange[0][0],simRange[0][1])
	# ax.set_ylim(simRange[1][0],simRange[1][1])
	# ax.set_zlim(simRange[2][0],simRange[2][1])
	# ax.plot(redPlaneData[0], redPlaneData[1], redPlaneData[2], color = 'red')
	# ax.plot(bluePlaneData[0], bluePlaneData[1], bluePlaneData[2], color = 'blue')
	# plt.xlabel('Lon')
	# plt.ylabel('Lat')
	# plt.savefig('trajectory.jpg')
	# plt.show()

	# visual = Visualization(redPlaneData, bluePlaneData, simRange=simRange)
	# ani = FuncAnimation(visual.fig, visual.update, frames = np.arange(timeSpan), interval = 50)
	# ani.save("move.gif", writer = 'Pillow', fps = 10)
	# plt.show()