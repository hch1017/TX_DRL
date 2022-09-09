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
		self.m_AircraftMoveInfo = SAircraftMoveInfo_AI_Interface()
		self.m_RadarInfo =SRadarInfo_AI_Interface()
		self.m_HMDInfo =SHMDInfo_AI_Interface()
		self.m_FCInfo =SFCInfo_AI_Interface()
		self.m_ESMInfo =SESMInfo_AI_Interface()
		self.m_DASInfo =SDASInfo_AI_Interface()
		self.m_AWACSInfo =SAWACSInfo_AI_Interface()
		self.m_MRAAMDataMsgSet =SMRAAMDataMsgSet_AI_Interface()
		self.m_AAMDataSet =SAAMDataSet_AI_Interface()

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
# 		self.m_eRadarOnOff = Enum_RadarOnOff_OFF
		self.m_eEleScanLine	= Enum_RadarEleScanLine_2
		self.m_eAziScanRange = Enum_RadarAziScanRange_30
# 		self.m_fAziScanCent	= 0
# 		self.m_fEleScanCent	= 0

class SWeaponCtrlCmd_AI_Interface():
	def __init__(self):
		self.m_bWeaponLaunch = False

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

	input.m_AircraftBasicInfo.m_ID = next(it)
	input.m_AircraftBasicInfo.m_lTimeStamp = next(it)
	input.m_AircraftBasicInfo.m_bAlive = next(it)
	input.m_AircraftBasicInfo.m_fFuel = next(it)

	input.m_AircraftMoveInfo.m_dSelfLon = next(it)
	input.m_AircraftMoveInfo.m_dSelfLat = next(it)
	input.m_AircraftMoveInfo.m_dSelfAlt = next(it)
	input.m_AircraftMoveInfo.m_fVN = next(it)
	input.m_AircraftMoveInfo.m_fVU = next(it)
	input.m_AircraftMoveInfo.m_fVE = next(it)
	input.m_AircraftMoveInfo.m_fAccN = next(it)
	input.m_AircraftMoveInfo.m_fAccU = next(it)
	input.m_AircraftMoveInfo.m_fAccE = next(it)
	input.m_AircraftMoveInfo.m_fAccBX = next(it)
	input.m_AircraftMoveInfo.m_fAccBY = next(it)
	input.m_AircraftMoveInfo.m_fAccBZ = next(it)
	input.m_AircraftMoveInfo.m_fTASpeed = next(it)
	input.m_AircraftMoveInfo.m_fMach = next(it)
	input.m_AircraftMoveInfo.m_fNormalAccel = next(it)
	input.m_AircraftMoveInfo.m_fYaw = next(it)
	input.m_AircraftMoveInfo.m_fPitch = next(it)
	input.m_AircraftMoveInfo.m_fRoll = next(it)
	input.m_AircraftMoveInfo.m_fAlpha = next(it)
	input.m_AircraftMoveInfo.m_fBeta = next(it)
	input.m_AircraftMoveInfo.m_fCrab = next(it)
	input.m_AircraftMoveInfo.m_fOmegaYaw = next(it)
	input.m_AircraftMoveInfo.m_fOmegaPitch = next(it)
	input.m_AircraftMoveInfo.m_fOmegaRoll = next(it)

	input.m_RadarInfo.m_eRadarWorkMode = next(it)
	input.m_RadarInfo.m_fAziScanRange = next(it)
	input.m_RadarInfo.m_fEleScanRange = next(it)
	input.m_RadarInfo.m_fAziScanCent = next(it)
	input.m_RadarInfo.m_fEleScanCent = next(it)

	input.m_RadarInfo.m_uiTgtNum = next(it)
	for radar_num in range(MAX_TGT_NUM):
		input.m_RadarInfo.m_TgtInfo[radar_num].m_uiTgtLot = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtDis = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAzi = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtEle = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVN = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVU = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtVE = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccN = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccU = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtAccE = next(it)
		input.m_RadarInfo.m_TgtInfo[radar_num].m_fTgtDisDot = next(it)

	input.m_HMDInfo.m_uiTgtNum = next(it)
	for hmd_num in range(MAX_TGT_NUM):
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_uiTgtLot = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtDis = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAziC = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtEleC = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fYaw = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fPitch = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fRoll = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVN = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVU = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtVE = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccN = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccU = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtAccE = next(it)
		input.m_HMDInfo.m_TgtInfo[hmd_num].m_fTgtDisDot = next(it)

	input.m_FCInfo.m_eAircraftMainState = next(it)
	input.m_FCInfo.m_fRmax = next(it)
	input.m_FCInfo.m_fRnoescape = next(it)
	input.m_FCInfo.m_fRmin = next(it)
	input.m_FCInfo.m_bINRNG = next(it)
	input.m_FCInfo.m_bSHOOT = next(it)
	input.m_FCInfo.m_bWeaponReady = next(it)

	input.m_ESMInfo.m_uiAlarmTgtNum = next(it)
	for esm_num in range(MAX_TGT_NUM):
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_uiTgtLot = next(it)
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_eTgtType = next(it)
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_eTgtIff = next(it)
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_fTgtAzi = next(it)
		input.m_ESMInfo.m_AlarmTgtInfo[esm_num].m_fTgtEle = next(it)

	input.m_DASInfo.m_uiThreatTgtNum = next(it)
	for das_num in range(MAX_TGT_NUM):
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_uiTgtLot = next(it)
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_fTgtAzi = next(it)
		input.m_DASInfo.m_ThreatTgtInfo[das_num].m_fTgtEle = next(it)

	input.m_AWACSInfo.m_uiTgtNum = next(it)
	for awacs_num in range(MAX_TGT_NUM):
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_uiTgtLot = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_eIFF = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLon = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtLat = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtAlt = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVN = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVU = next(it)
		input.m_AWACSInfo.m_TgtInfo[awacs_num].m_fTgtVE = next(it)

	input.m_MRAAMDataMsgSet.m_uiMsgNum = next(it)
	for mraamd_num in range(MAX_MRAAM_NUM):
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_uiAAMID = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_bSeekerOpen = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_bCapture = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLon = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dLat = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_dAlt = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVX = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVU = next(it)
		input.m_MRAAMDataMsgSet.m_MRAAMDataMsg[mraamd_num].m_fMslVE = next(it)

	input.m_AAMDataSet.m_iAAMNum = next(it)
	for aamd_num in range(MAX_AAM_NUM_IN_SCENE):
		input.m_AAMDataSet.m_AAMData[aamd_num].m_uiAAMID = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_eAAMType = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_uiPlaneID = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_eAAMState = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_bSeekerOpen = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_bCapture = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLon = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dLat = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_dAlt = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVX = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVU = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslVE = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslYaw = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fMslPitch = next(it)
		input.m_AAMDataSet.m_AAMData[aamd_num].m_fTgtDis = next(it)

	output.m_FlyCtrlCmd.m_fStickLat = next(it)
	output.m_FlyCtrlCmd.m_fStickLon = next(it)
	output.m_FlyCtrlCmd.m_fThrottle = next(it)
	output.m_FlyCtrlCmd.m_fRudder = next(it)
	output.m_FCCtrlCmd.m_eMainTaskMode = next(it)
# 	output.m_FCCtrlCmd.m_eRadarOnOff = next(it)
	output.m_RadarCtrlCmd.m_eEleScanLine = next(it)
	output.m_RadarCtrlCmd.m_eAziScanRange = next(it)
# 	output.m_RadarCtrlCmd.m_fAziScanCent = next(it)
# 	output.m_RadarCtrlCmd.m_fEleScanCent = next(it)
	output.m_WeaponCtrlCmd.m_bWeaponLaunch = next(it)
	return input, output

if __name__ == '__main__':
	AIPilotInit()
	AIPilotInput()
	AIPilotOutput()