import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  AudioModule,
  createAudioPlayer,
  RecordingPresets,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';

import { File, Paths } from 'expo-file-system';

import { CameraView, useCameraPermissions } from 'expo-camera';

import * as ImageManipulator from 'expo-image-manipulator';
import { router } from 'expo-router';
import * as Speech from 'expo-speech';
import { fetch } from 'expo/fetch';
import { FlatList } from 'react-native';

import * as ImagePicker from 'expo-image-picker';
import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Dimensions,
  Easing,
  ImageBackground,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

// AI 첫 발화 타이핑 애니메이션

// AI 첫 발화 글자 순차 표시


const { width, height } = Dimensions.get('window');
const BASE_URL = 'https://cobalt-unretired-fastness.ngrok-free.dev';

// 백엔드가 꺼져있을 때는 true
const USE_MOCK = false;

// 파동 개수 (12개)
const WAVE_BAR_COUNT = 12;

// =====================================
// 방향별 객체 중심 좌표 변환 (resizeMode="cover" 기준)
// areaWidth/areaHeight: 사진이 실제로 그려지는 영역 크기 (onLayout으로 측정)
// =====================================
const DIRECTIONS = ['front', 'right', 'back', 'left'];

function convertViewsToDots(views, areaWidth, areaHeight) {
  const result = {};

  DIRECTIONS.forEach((direction) => {
    const objects = views?.[direction]?.objects || [];
    const imageWidth = Number(views?.[direction]?.image_size?.width);
    const imageHeight = Number(views?.[direction]?.image_size?.height);

    if (!imageWidth || !imageHeight || !areaWidth || !areaHeight) {
      result[direction] = [];
      return;
    }

    const scale = Math.max(areaWidth / imageWidth, areaHeight / imageHeight);
    const offsetX = (areaWidth - imageWidth * scale) / 2;
    const offsetY = (areaHeight - imageHeight * scale) / 2;

    result[direction] = objects
      .map((obj, index) => {
        if (!obj?.box) return null;

        const { x1, y1, x2, y2 } = obj.box;

        // YOLO Bounding Box 중앙 좌표 → 화면 좌표
        const x = ((Number(x1) + Number(x2)) / 2) * scale + offsetX;
        const y = ((Number(y1) + Number(y2)) / 2) * scale + offsetY;

        // 화면 밖의 객체 제외
        if (x < 0 || x > areaWidth || y < 0 || y > areaHeight) {
          return null;
        }

        return {
          id: obj.id || `${direction}_${index}`,
          name: obj.name || obj.label || 'Object',
          confidence: obj.confidence ?? 0,
          x,
          y,
          box: obj.box,
          direction,
        };
      })
      .filter(Boolean);
  });

  return result;
}

export default function CameraScreen({ onSaveToFeedback }) {

const [openingDisplayText, setOpeningDisplayText] = useState('');
const openingTypingRef = useRef(null);
const audioPlayerRef = useRef(null);

const animateOpeningText = (text, duration) => {
  if (openingTypingRef.current) {
    clearInterval(openingTypingRef.current);
    openingTypingRef.current = null;
  }

  setOpeningDisplayText('');

  let index = 0;

  // 전체 문장 길이에 맞춰 글자 출력 간격 계산
  const interval = duration / text.length;

  openingTypingRef.current = setInterval(() => {
    index++;

    setOpeningDisplayText(text.slice(0, index));

    if (index >= text.length) {
      clearInterval(openingTypingRef.current);
      openingTypingRef.current = null;
    }
  }, interval);
};

const audioRecorder = useAudioRecorder(
  RecordingPresets.HIGH_QUALITY
);

const recorderState = useAudioRecorderState(audioRecorder);


  const userId = 1;
  const cameraRef = useRef(null);
  const [permission, requestPermission] = useCameraPermissions();
  

  // 화면 단계: 'CAMERA' | 'ANALYZING' | 'MAIN_LEARNING' | 'CHAT'
  const [step, setStep] = useState('CAMERA');
const [capturedPhotoUri, setCapturedPhotoUri] = useState(null);
const [capturedPhotoUris, setCapturedPhotoUris] = useState([]);
const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0);
const [directionObjects, setDirectionObjects] = useState({});
// 백엔드 views 원본 (image_size, box 포함)
const [rawViews, setRawViews] = useState(null);
// 사진이 실제로 그려지는 영역 크기 (FlatList onLayout으로 측정)
const [photoArea, setPhotoArea] = useState({ width, height });

// 사진 영역 크기가 측정되거나 바뀌면 점 위치를 다시 계산
useEffect(() => {
  if (!rawViews) return;

  const dots = convertViewsToDots(rawViews, photoArea.width, photoArea.height);
  setDirectionObjects(dots);
  setDetectedObjects(dots[DIRECTIONS[currentPhotoIndex]] || []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [rawViews, photoArea.width, photoArea.height]);

const PHOTO_DIRECTIONS = ['앞쪽', '오른쪽', '뒤쪽', '왼쪽'];

// 4방향 촬영
const [capturedPhotos, setCapturedPhotos] = useState({
  front: null,
  right: null,
  back: null,
  left: null,
});

// 현재 촬영할 방향
const [captureDirection, setCaptureDirection] = useState('front');
const [isCameraReady, setIsCameraReady] = useState(false);
const [isTakingPicture, setIsTakingPicture] = useState(false);

// 4장 촬영이 끝났는지
const CAPTURE_DIRECTIONS = [
  { key: 'front', label: '앞쪽', instruction: '앞쪽 공간을 촬영해주세요' },
  { key: 'right', label: '오른쪽', instruction: '오른쪽 공간을 촬영해주세요' },
  { key: 'back', label: '뒤쪽', instruction: '뒤쪽 공간을 촬영해주세요' },
  { key: 'left', label: '왼쪽', instruction: '왼쪽 공간을 촬영해주세요' },
];
  // 인식 및 상태 관리
  const [selectedObjectId, setSelectedObjectId] = useState(null);
const [currentSpace, setCurrentSpace] = useState('');
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [selectedObject, setSelectedObject] = useState(null);
  const [recommendedTopics, setRecommendedTopics] = useState([]);

  // UI 모달 및 피드백 버튼 노출 상태
  const [isTopicDrawerOpen, setIsTopicDrawerOpen] = useState(false);
  const [showFeedbackBtn, setShowFeedbackBtn] = useState(false);
  const [showRecommendedTopics, setShowRecommendedTopics] = useState(false);

  // 대화 및 AI 상태
  const [isStartingConversation, setIsStartingConversation] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [activeDialogue, setActiveDialogue] = useState([]);
  const [aiStatus, setAiStatus] = useState('IDLE');
  const [currentTopic, setCurrentTopic] = useState('');
  const [contextId, setContextId] = useState(null);
  const scrollViewRef = useRef(null);
  // 🎙️ 음성 녹음/인식 상태: 'IDLE' | 'RECORDING' | 'PROCESSING'
  const [recordingState, setRecordingState] = useState('IDLE');

// AI 응답 대기 점 애니메이션
const thinkingDots = useRef([
  new Animated.Value(0),
  new Animated.Value(0),
  new Animated.Value(0),
]).current;

useEffect(() => {
  if (
  aiStatus !== 'THINKING' &&
  recordingState !== 'RECORDING'
) {
  thinkingDots.forEach(dot => dot.setValue(0));
  return;
}

  const [dot1, dot2, dot3] = thinkingDots;

  const animation = Animated.loop(
    Animated.sequence([
      Animated.timing(dot1, {
        toValue: 1,
        duration: 180,
        useNativeDriver: false,
      }),
      Animated.timing(dot2, {
        toValue: 1,
        duration: 180,
        useNativeDriver: false,
      }),
      Animated.timing(dot3, {
        toValue: 1,
        duration: 180,
        useNativeDriver: false,
      }),
      Animated.parallel(
        thinkingDots.map(dot =>
          Animated.timing(dot, {
            toValue: 0,
            duration: 180,
            useNativeDriver: false,
          })
        )
      ),
    ])
  );

  animation.start();

  return () => animation.stop();
}, [aiStatus]);

  // 🌊 음성 파동 애니메이션
  const waveAnimValues = useRef(
    Array.from({ length: WAVE_BAR_COUNT }, () => new Animated.Value(6))
  ).current;

  // 패러프레이징 토글 상태
  const [showParaphraseMap, setShowParaphraseMap] = useState({});

  // 예: 주제를 선택하거나 대화 화면으로 전환되는 함수
const startChat = (topic) => {
  setCurrentTopic(topic);
  
  // 타이머
  setTimeLeft(40); 
  setShowFeedbackBtn(false);
  
  setStep('CHAT'); // 화면을 CHAT 단계로 전환 (이제 useEffect 타이머 동작!)
};
const sendVoiceState = async (event) => {
  try {
    const response = await fetch(
      `${BASE_URL}/voice/conversation/${sessionId}/state`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({ event }),
      }
    );

    const data = await response.json();

    console.log(`📡 음성 상태 전송 (${event}):`, data);

    return data;

  } catch (error) {
    console.error(`❌ 음성 상태 전송 실패 (${event}):`, error);
    throw error;
  }
};

const startRecording = async () => {
  try {
    if (recordingState !== 'IDLE') return;
    if (aiStatus === 'THINKING') return;

    console.log('🎙️ 녹음 시작');

    await sendVoiceState('user_speech_started');

    // 이미 녹음 중이거나 준비 중인지 확인
    if (recorderState.isRecording) {
      console.log('이미 녹음 중입니다.');
      return;
    }

    await audioRecorder.prepareToRecordAsync();
    audioRecorder.record();
    setActiveDialogue(prev => [
  ...prev.filter(chat => !chat.isUserThinking),
  {
    id: `user_thinking_${Date.now()}`,
    speaker: 'USER',
    isUserThinking: true,
  },
]);

    setRecordingState('RECORDING');
    console.log('🔴 녹음 중...');
  } catch (error) {
    console.error('❌ 녹음 시작 실패:', error);
    setRecordingState('IDLE');
    setAiStatus('IDLE');

    alert('음성 녹음을 시작할 수 없습니다.');
  }
};
const stopRecording = async () => {
  try {
    console.log('🛑 녹음 종료');

    setRecordingState('PROCESSING');
    setAiStatus('THINKING');

    await audioRecorder.stop();

    const audioUri = audioRecorder.uri;

    console.log('🎧 audioUri:', audioUri);

    if (!audioUri) {
      throw new Error('녹음 파일을 가져오지 못했습니다.');
    }

    // 1. 음성 파일을 백엔드에 전송하고 AI 응답까지 대기
    await sendAudioToBackend(audioUri);

    // 2. 사용자 발화 처리 완료 알림
    await sendVoiceState('user_speech_finished');

  } catch (error) {
    console.error('❌ 녹음 종료/전송 실패:', error);

    setRecordingState('IDLE');
    setAiStatus('IDLE');

    alert(
      error?.message ||
      '음성 처리 중 오류가 발생했습니다.'
    );
  }
};


  //타이머
const [timeLeft, setTimeLeft] = useState(30);

useEffect(() => {
  let timer;

  // step이 'CHAT'일 때만 타이머 동작
  if (step === 'CHAT') {
    if (timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
    } else if (timeLeft === 0) {
      // 60초가 모두 끝나면 피드백 버튼 노출
      setShowFeedbackBtn(true);
    }
  }

  return () => clearInterval(timer);
}, [step, timeLeft]);

  // 🌊 흘러가는 음성 파동 애니메이션
  useEffect(() => {
    let animations = [];

    if (recordingState === 'RECORDING') {
      animations = waveAnimValues.map((anim, index) => {
        const baseMin = 8;
        const baseMax = 28 + (index % 3) * 4;

        return Animated.loop(
          Animated.sequence([
            Animated.timing(anim, {
              toValue: baseMax,
              duration: 250 + (index % 4) * 50,
              easing: Easing.inOut(Easing.sin),
              useNativeDriver: false,
            }),
            Animated.timing(anim, {
              toValue: baseMin,
              duration: 250 + (index % 4) * 50,
              easing: Easing.inOut(Easing.sin),
              useNativeDriver: false,
            }),
          ])
        );
      });

      animations.forEach((anim) => anim.start());
    } else {
      waveAnimValues.forEach((anim) => {
        Animated.timing(anim, {
          toValue: 6,
          duration: 300,
          useNativeDriver: false,
        }).start();
      });
    }

    return () => {
      animations.forEach((anim) => anim.stop());
    };
  }, [recordingState]);




  useEffect(() => {
  const setupAudio = async () => {
    try {
      const permission =
        await AudioModule.requestRecordingPermissionsAsync();

      if (!permission.granted) {
        alert('마이크 권한이 필요합니다.');
        return;
      }

      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });

      console.log('🎙️ 마이크 권한 및 오디오 설정 완료');
    } catch (error) {
      console.error(
        '❌ 오디오 설정 오류:',
        error
      );
    }
  };

  setupAudio();
}, []);


  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // 1. 이미지 통합 분석
// 1. 이미지 통합 분석
const processImageAnalysis = async (photoUri) => {
  setStep('ANALYZING');

  const formattedUri =
    Platform.OS === 'ios'
      ? photoUri.replace('file://', '')
      : photoUri;

  try {
    const formData = new FormData();

    // 💡 단건 분석도 File 객체 규격으로 생성
    const imageFile = new File(formattedUri);
    formData.append('file', imageFile);

    formData.append('user_id', String(userId));

    const response = await fetch(
      `${BASE_URL}/detect-space`,
      {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      }
    );


console.log(
  '📡 이미지 분석 응답 상태:',
  response.status
);

console.log(
  '📡 이미지 분석 응답 URL:',
  response.url
);


const responseText = await response.text();

const { audio_base64, ...logData } = data;

console.log('📦 서버 응답:', logData);


let data;

try {
  data = JSON.parse(responseText);
} catch (parseError) {
  console.error(
    `❌ ${direction} 응답이 JSON이 아닙니다:`,
    responseText
  );

  throw new Error(
    `${direction} 이미지 분석 서버가 JSON이 아닌 응답을 반환했습니다.`
  );
}

    console.log('📦 detect-space 전체 응답:', JSON.stringify(data, null, 2));

    // ==========================================
    // 1. 공간 정보
    // ==========================================
    const detectedSpace =
      data?.space ||
      data?.place ||
      data?.detected_space ||
      '공간';

    setCurrentSpace(detectedSpace);

    // ==========================================
// 2. 사물 + Bounding Box
// ==========================================
if (Array.isArray(data?.objects)) {

const YOLO_WIDTH = data.image_size?.width;
const YOLO_HEIGHT = data.image_size?.height;

if (!YOLO_WIDTH || !YOLO_HEIGHT) {
  console.warn('⚠️ 이미지 크기 정보가 없습니다.', data.image_size);
  setDetectedObjects([]);
  return;
}

  const mappedObjects = data.objects
    .map((obj, index) => {

      // Bounding Box 유효성 검사
      if (
        !obj?.box ||
        typeof obj.box.x1 !== 'number' ||
        typeof obj.box.y1 !== 'number' ||
        typeof obj.box.x2 !== 'number' ||
        typeof obj.box.y2 !== 'number'
      ) {
        console.warn('⚠️ Bounding Box 데이터 오류:', obj);
        return null;
      }

      const { x1, y1, x2, y2 } = obj.box;

     // Bounding Box 중심 좌표 계산
const centerX = (x1 + x2) / 2;
const centerY = (y1 + y2) / 2;

// YOLO 좌표 → 화면 좌표 변환
const x = (centerX / YOLO_WIDTH) * width;
const y = (centerY / YOLO_HEIGHT) * height;

const boxWidth = ((x2 - x1) / YOLO_WIDTH) * width;
const boxHeight = ((y2 - y1) / YOLO_HEIGHT) * height;

      // 잘못된 크ㄴ기의 박스 제외
      if (boxWidth <= 0 || boxHeight <= 0) {
        console.warn('⚠️ Bounding Box 크기 오류:', obj);
        return null;
      }

      const mappedObject = {
        id: obj.id || `obj_${index}`,
        name: obj.name || obj.label || 'Object',
        confidence: obj.confidence ?? 0,

        x,
        y,
        w: boxWidth,
        h: boxHeight,

        // 원본 YOLO 좌표 보관
        box: obj.box,
      };

      console.log(`📍 ${mappedObject.name}:`, {
        original: obj.box,
        screen: {
        centerX: x,
        centerY: y,
        width: boxWidth,
        height: boxHeight,
  },
});

      return mappedObject;
    })
    .filter(Boolean);

  console.log(
    '📍 최종 화면 객체:',
    JSON.stringify(mappedObjects, null, 2)
  );

  setDetectedObjects(mappedObjects);

} else {
  console.warn('⚠️ objects가 없습니다.');
  setDetectedObjects([]);
}
    // ==========================================
    // 3. 추천 표현 / 추천 주제
    // ==========================================
    const topics =
      data?.ai?.expressions ||
      data?.expressions ||
      data?.recommended_topics ||
      [];

    setRecommendedTopics(
      Array.isArray(topics) && topics.length > 0
        ? topics
        : DUMMY_TOPICS
    );

    // ==========================================
    // 학습 화면으로 이동
    // ==========================================
    console.log('🚀 MAIN_LEARNING 화면으로 이동');
setStep('MAIN_LEARNING');

  } catch (err) {
    console.error('이미지 분석 통신 에러:', err);

    alert('공간 및 사물 인식 중 오류가 발생했습니다.');

    setStep('CAMERA');
  }
};



// ==========================================
// 4방향 촬영 버튼
// ==========================================
const handleTakePicture = async () => {
  if (!cameraRef.current) {
    console.warn('⚠️ 카메라가 준비되지 않았습니다.');
    return;
  }

  try {
    console.log(
      `📸 ${CAPTURE_DIRECTIONS.find(
        (item) => item.key === captureDirection
      )?.label} 촬영 시작`
    );

    // 사진 촬영
    const photo = await cameraRef.current.takePictureAsync({
  quality: 0.6,
  skipProcessing: true,
});

if (!photo?.uri) {
  console.warn('⚠️ 사진 URI가 없습니다.');
  return;
}

console.log('✅ 원본 사진 촬영 성공:', photo.uri);

// 📐 사진 크기 줄이기
const resizedPhoto = await ImageManipulator.manipulateAsync(
  photo.uri,
  [
    {
      resize: {
        width: 1000,
      },
    },
  ],
  {
    compress: 0.6,
    format: ImageManipulator.SaveFormat.JPEG,
  }
);

console.log('✅ 리사이즈 완료:', resizedPhoto.uri);

const currentIndex = CAPTURE_DIRECTIONS.findIndex(
  (item) => item.key === captureDirection
);

// 현재 방향 사진 저장
const updatedPhotos = {
  ...capturedPhotos,
  [captureDirection]: resizedPhoto.uri,
};

setCapturedPhotos(updatedPhotos);

    // ==========================================
    // 아직 4장이 안 끝났으면 다음 방향으로 이동
    // ==========================================
    if (currentIndex < CAPTURE_DIRECTIONS.length - 1) {
      const nextDirection =
        CAPTURE_DIRECTIONS[currentIndex + 1];

      console.log(
        `➡️ 다음 방향: ${nextDirection.label}`
      );

      // 잠깐 기다린 뒤 방향 변경
      setTimeout(() => {
        setCaptureDirection(nextDirection.key);
      }, 300);

      return;
    }

    // ==========================================
    // 4장 모두 촬영 완료
    // ==========================================
    console.log('🎉 4방향 촬영 완료');

    setCapturedPhotoUri(updatedPhotos.front);

    // 분석 시작
    await processFourImages(updatedPhotos);

  } catch (error) {
    console.error('📸 촬영 오류:', error);
  }

};
const handlePickImages = async () => {
  try {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (!permissionResult.granted) {
      alert('갤러리 접근 권한이 필요합니다.');
      return;
    }

   const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'], // 👈 이렇게 문자열 배열이나 새로운 MediaType으로 변경
      allowsMultipleSelection: true,
      selectionLimit: 4,
      quality: 0.8,
    });

    if (result.canceled) {
      console.log('🖼️ 갤러리 선택 취소됨');
      return;
    }

    const selectedAssets = result.assets;

    if (!selectedAssets || selectedAssets.length < 4) {
      alert('4방향 분석을 위해 정확히 4장의 사진을 선택해주세요.');
      return;
    }

    console.log(`✅ 갤러리에서 사진 ${selectedAssets.length}장 선택됨`);

    setStep('ANALYZING');

    const processedPhotos = {};
    const directions = ['front', 'right', 'back', 'left'];

    for (let i = 0; i < 4; i++) {
      const asset = selectedAssets[i];
      const directionKey = directions[i];

      const formattedUri =
        Platform.OS === 'ios' && asset.uri.startsWith('file://')
          ? asset.uri.replace('file://', '')
          : asset.uri;

      const resizedPhoto = await ImageManipulator.manipulateAsync(
        formattedUri,
        [{ resize: { width: 1000 } }],
        {
          compress: 0.6,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );

      processedPhotos[directionKey] = resizedPhoto.uri;
    }

    console.log('✅ 4장 리사이즈 및 방향 매핑 완료:', processedPhotos);

    setCapturedPhotos(processedPhotos);
    await processFourImages(processedPhotos);

  } catch (error) {
    console.error('🖼️ 갤러리 선택 오류:', error);
    alert('갤러리에서 사진을 불러오는 중 오류가 발생했습니다.');
    setStep('CAMERA');
  }
};

const processFourImages = async (photos) => {
  setStep('ANALYZING');

  try {
    if (
      !photos.front ||
      !photos.right ||
      !photos.back ||
      !photos.left
    ) {
      throw new Error('4장 사진이 모두 필요합니다.');
    }

    console.log('🚀 4방향 이미지 분석 시작');

    const formData = new FormData();

    // ✅ 성공했던 File 객체 활용 방식
    const frontFile = new File(photos.front);
    const rightFile = new File(photos.right);
    const backFile = new File(photos.back);
    const leftFile = new File(photos.left);

    formData.append('front', frontFile);
    formData.append('right', rightFile);
    formData.append('back', backFile);
    formData.append('left', leftFile);

    formData.append('user_id', String(userId));

    console.log('📤 /detect-space-4 전송 시작');

    const response = await fetch(
      `${BASE_URL}/detect-space-4`,
      {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      }
    );


    console.log('📡 4방향 응답 상태:', response.status);

    const data = await response.json();

    console.log('📦 4방향 분석 결과:', JSON.stringify(data, null, 2));

    if (!response.ok || !data.success) {
      throw new Error(
        data.message || '4방향 이미지 분석에 실패했습니다.'
      );
    }

// 백엔드가 반환한 공간 인식 ID 저장
setContextId(data.contextId ?? null);

console.log('📌 저장된 contextId:', data.contextId);

    // =====================================
    // 사진 순서 저장
    // front → right → back → left
    // =====================================

    setCapturedPhotoUris([
      photos.front,
      photos.right,
      photos.back,
      photos.left,
    ]);

    setCurrentPhotoIndex(0);

    // 기존 코드와의 호환
    setCapturedPhotoUri(photos.front);

    // =====================================
    // 방향별 분석 결과
    // =====================================

    const views = data.views || {};

    console.log('앞:', views.front);
    console.log('오른쪽:', views.right);
    console.log('뒤:', views.back);
    console.log('왼쪽:', views.left);

// =====================================
// 방향별 객체 좌표 변환은 아래 useEffect에서
// 실제 사진 영역 크기(photoArea)를 기준으로 수행합니다.
// =====================================
const initialDots = convertViewsToDots(views, photoArea.width, photoArea.height);
setRawViews(views);
setDirectionObjects(initialDots);
setDetectedObjects(initialDots.front || []);

setSelectedObject(null);

// ⭐⭐⭐ 이 부분은 반드시 그대로 아래에 있어야 함

const space =
  views.front?.space ||
  views.front?.place ||
  views.right?.space ||
  views.right?.place ||
  views.back?.space ||
  views.back?.place ||
  views.left?.space ||
  views.left?.place ||
  '알 수 없는 공간';

setCurrentSpace(space);

console.log('🏠 현재 공간:', space);

// =====================================
// 추천 주제
// =====================================

// 백엔드 추천 주제 계산 구조는 유지
const recommendedTopicsFromBackend = [];

Object.values(views).forEach((view) => {
  if (view?.scores) {
    Object.entries(view.scores).forEach(
      ([topic, score]) => {
        recommendedTopicsFromBackend.push({
          topic,
          score,
        });
      }
    );
  }
});

recommendedTopicsFromBackend.sort(
  (a, b) => b.score - a.score
);

const uniqueTopics = [];
const seenTopics = new Set();

recommendedTopicsFromBackend.forEach((item) => {
  if (!seenTopics.has(item.topic)) {
    seenTopics.add(item.topic);
    uniqueTopics.push(item);
  }
});


    // 백엔드에서 반환한 장소별 추천 주제 사용
    const topics = Array.isArray(data?.topics)
      ? data.topics
      : [];

    setRecommendedTopics(topics);

    console.log('🎯 추천 주제:', topics);


    // =====================================
    // 메인 학습 화면
    // =====================================

    console.log('🎉 4장 이미지 분석 전체 완료');

    setStep('MAIN_LEARNING');

  } catch (err) {
    console.error(
      '❌ 4방향 이미지 분석 에러:',
      err
    );

    alert(
      err.message ||
      '4방향 공간 및 사물 인식 중 오류가 발생했습니다.'
    );

    setStep('CAMERA');
  }
};

// ==========================================
// 4방향 분석 결과 통합
// ==========================================
const mergeFourImageResults = (results, photos) => {
  console.log('🔄 4방향 분석 결과 통합 시작');

  let mergedSpace = '';
  let mergedObjects = [];
  let mergedTopics = [];

  results.forEach((result) => {
    const data = result.data;
    const direction = result.direction;

    // =====================================
    // 1. 공간 정보
    // =====================================
    if (!mergedSpace) {
      mergedSpace =
        data?.space ||
        data?.place ||
        data?.detected_space ||
        '';
    }


// 2. 객체 정보
// =====================================
if (Array.isArray(data?.objects)) {
  const directionObjects = data.objects.map((obj, index) => {
      // Bounding Box 확인
      if (
        !obj?.box ||
        typeof obj.box.x1 !== 'number' ||
        typeof obj.box.y1 !== 'number' ||
        typeof obj.box.x2 !== 'number' ||
        typeof obj.box.y2 !== 'number'
      ) {
        console.warn(
          `⚠️ ${direction} Bounding box가 없는 객체:`,
          obj
        );
        return null;
      }

  // =====================================
// YOLO Bounding Box (x1, y1, x2, y2)
// =====================================
const { x1, y1, x2, y2 } = obj.box;

// Bounding Box의 좌상단, 너비, 높이
const boxLeft = x1;
const boxTop = y1;
const boxWidth = x2 - x1;
const boxHeight = y2 - y1;

// =====================================
// 백엔드가 보내준 실제 이미지 크기
// =====================================
const imageWidth = data?.image_size?.width || 990;
const imageHeight = data?.image_size?.height || 1920;

// =====================================
// YOLO 원본 좌표 → 모바일 화면 스케일링 좌표
// =====================================
const screenX = (boxLeft / imageWidth) * width;
const screenY = (boxTop / imageHeight) * height;
const screenW = (boxWidth / imageWidth) * width;
const screenH = (boxHeight / imageHeight) * height;

console.log(`📍 ${direction.toUpperCase()} 객체 바운딩박스 좌표:`, {
  name: obj.name || obj.label,
  screen: {
    x: screenX,
    y: screenY,
    w: screenW,
    h: screenH,
  },
});

return {
  id: `${direction}_${obj.id || index}`,

  name: obj.name || obj.label || 'Object',

  confidence: obj.confidence || 0,

  // 바운딩 박스용 화면 변환 값 전달
  x: screenX,
  y: screenY,
  w: screenW,
  h: screenH,

  box: obj.box,

  // ⭐ 이 객체가 어느 사진에 있는지 저장
  direction,
};
    })
    .filter(Boolean);

  mergedObjects.push(...directionObjects);
}

    // =====================================
    // 3. 추천 주제
    // =====================================
    const topics =
      data?.ai?.expressions ||
      data?.expressions ||
      data?.recommended_topics ||
      [];

    if (Array.isArray(topics)) {
      mergedTopics.push(...topics);
    }
  });

  // =====================================
  // 4. 추천 주제 중복 제거
  // =====================================
  mergedTopics = [...new Set(mergedTopics)];

  console.log('🏠 통합 공간:', mergedSpace);
  console.log(
    '📦 FRONT 객체:',
    mergedObjects
  );
  console.log(
    '💡 통합 주제:',
    mergedTopics
  );

  // =====================================
  // 5. State 저장
  // =====================================
  setCurrentSpace(
    mergedSpace || '공간'
  );

  setDetectedObjects(
    mergedObjects
  );

  setRecommendedTopics(
    mergedTopics.length > 0
      ? mergedTopics
      : DUMMY_TOPICS
  );

  // =====================================
  // 6. 대표 사진 = 앞쪽 사진
  // =====================================
  if (photos?.front) {
    setCapturedPhotoUri(
      photos.front
    );
  }

  // =====================================
  // 7. 학습 화면으로 이동
  // =====================================
  setStep('MAIN_LEARNING');
};

  const handlePlayPronunciation = (word) => {
    if (Speech) Speech.speak(word, { language: 'en-US' });
  };
const startFreeConversationAPI = async () => {
  // 중복 요청 방지
  if (isStartingConversation) return;

  console.log('🚀 자유주제 대화 시작');

  setIsStartingConversation(true);
  setCurrentTopic('자유주제');
  setShowFeedbackBtn(false);
  setActiveDialogue([]);
  setShowParaphraseMap({});
  setRecordingState('IDLE');
  setAiStatus('THINKING');
  setTimeLeft(40);

  try {
    const formData = new FormData();

formData.append('user_id', String(userId));
formData.append('contextId', String(contextId));
formData.append('mode', 'free');

// 자유주제이므로 topic_id는 보내지 않음
    const response = await fetch(
      `${BASE_URL}/voice/conversation/start`,
      {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
        body: formData,
      }
    );

    const data = await response.json();

    console.log(
      '📡 자유주제 시작 응답:',
      JSON.stringify(data, null, 2)
    );

    if (!response.ok || !data?.success) {
      throw new Error(
        data?.message ||
        data?.detail ||
        `HTTP ${response.status}`
      );
    }

    const newSessionId =
      data?.session_id ||
      data?.sessionId ||
      data?.session?.id;

    if (!newSessionId) {
      throw new Error('session_id를 받지 못했습니다.');
    }

    setSessionId(newSessionId);

// 4. AI 첫 인사 준비

// AI 첫 인사 준비
const openingText = data?.ai_text || '';

// 기존 애니메이션이 실행 중이면 중지
if (openingTypingRef.current) {
  clearInterval(openingTypingRef.current);
  openingTypingRef.current = null;
}

// 첫 인사 글자 초기화
setOpeningDisplayText('');

// 첫 인사 말풍선 등록
setActiveDialogue(
  openingText
    ? [{
        id: `ai_opening_${Date.now()}`,
        speaker: 'AI',
        text: openingText,
        isOpening: true,
      }]
    : []
);

// 대화 화면으로 이동
setStep('CHAT');

// 첫 인사 애니메이션 실행
if (openingText) {
  // 예상 음성 재생 시간 계산
  const estimatedDuration = (openingText.length / 9.5) * 1000;

  await Speech.stop();

  // 텍스트 애니메이션 실행 (한 번만 호출)
  animateOpeningText(openingText, estimatedDuration);

  // AI 음성 재생
  Speech.speak(openingText, {
    language: 'en-US',
    rate: 0.95,
    pitch: 1.0,

    onDone: () => {
      if (openingTypingRef.current) {
        clearInterval(openingTypingRef.current);
        openingTypingRef.current = null;
      }

      setOpeningDisplayText(openingText);
      setAiStatus('IDLE');
      setRecordingState('IDLE');
    },

    onError: (error) => {
      console.error('❌ 첫 인사 재생 오류:', error);

      if (openingTypingRef.current) {
        clearInterval(openingTypingRef.current);
        openingTypingRef.current = null;
      }

      setOpeningDisplayText(openingText);
      setAiStatus('IDLE');
      setRecordingState('IDLE');
    },
  });
} else {
  setAiStatus('IDLE');
  setRecordingState('IDLE');
}
} catch (error) {
  console.error('❌ 자유주제 대화 시작 실패:', error);

  alert(
    error?.message ||
    '자유주제 대화를 시작할 수 없습니다.'
  );

  setAiStatus('IDLE');
  setRecordingState('IDLE');

} finally {
  setIsStartingConversation(false);
}
};
  // 4. 대화 시작
  const playKokoroAudio = async (audioBase64) => {
  if (!audioBase64) {
    throw new Error('백엔드 음성 데이터가 없습니다.');
  }

  try {
    // 기존 음성 플레이어 정리
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.release();
      audioPlayerRef.current = null;
    }

    // Base64 WAV 파일 생성
    const audioFile = new File(
      Paths.cache,
      `kokoro-${Date.now()}.wav`
    );

    audioFile.create();

    await audioFile.write(audioBase64, {
      encoding: 'base64',
    });

    console.log('🎵 Kokoro WAV 파일 생성:', audioFile.uri);

    // 오디오 플레이어 생성
    const player = createAudioPlayer(audioFile.uri);

    audioPlayerRef.current = player;

    // 재생 완료까지 대기
    await new Promise((resolve, reject) => {
      let finished = false;

      const subscription = player.addListener(
        'playbackStatusUpdate',
        (status) => {
          if (finished) return;

          if (status.error) {
            finished = true;
            subscription.remove();
            reject(new Error(status.error));
            return;
          }

          if (status.didJustFinish) {
            finished = true;
            subscription.remove();
            resolve();
          }
        }
      );

      player.play();
    });

    console.log('🎵 Kokoro 음성 재생 완료');

  } finally {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.release();
      audioPlayerRef.current = null;
    }
  }
};

const startConversationAPI = async (topic) => {
    console.log('🔥 수정한 startConversationAPI 실행 확인');

  console.log('🚀 대화 시작');
  console.log('📌 선택한 주제:', topic);

  // 1. 주제와 공간 인식 정보 확인
  const selectedTopicId = topic?.topic_id;

  if (!contextId) {
    alert('공간 인식 정보가 없습니다. 공간을 다시 인식해 주세요.');
    console.error('❌ contextId가 없습니다.');
    return;
  }

  if (!selectedTopicId) {
    alert('선택한 주제의 topic_id가 없습니다.');
    console.error('❌ topic_id가 없습니다:', topic);
    return;
  }

  // 화면에 표시할 주제명
  const topicName =
    topic?.title_ko ||
    topic?.title_en ||
    topic?.title ||
    '';

  setCurrentTopic(topicName);
  setIsTopicDrawerOpen(false);
  setShowFeedbackBtn(false);
  setActiveDialogue([]);
  setShowParaphraseMap({});
  setRecordingState('IDLE');
  setAiStatus('THINKING');

  try {
    const formData = new FormData();

    // 2. 백엔드에서 요구하는 필드 전송
    formData.append('user_id', String(userId));
    formData.append('contextId', String(contextId));
    formData.append('topic_id', String(selectedTopicId));
    formData.append('mode', 'topic');

    console.log('📤 대화 시작 요청:', {
      user_id: userId,
      contextId,
      topic_id: selectedTopicId,
      mode: 'topic',
    });

    const response = await fetch(
      `${BASE_URL}/voice/conversation/start`,
      {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
        body: formData,
      }
    );

    const data = await response.json();

    console.log('📡 시작 응답 상태:', response.status);
console.log('🤖 백엔드 응답:', {
  ...data,
  audio_base64: data?.audio_base64 ? '[음성 데이터 생략]' : undefined,
});

    if (!response.ok || !data?.success) {
      throw new Error(
        data?.message ||
        data?.detail ||
        `HTTP ${response.status}`
      );
    }

    // 3. 세션 ID 확인
    const newSessionId =
      data?.session_id ||
      data?.sessionId ||
      data?.session?.id;

    if (!newSessionId) {
      throw new Error('session_id를 받지 못했습니다.');
    }

    setSessionId(newSessionId);

    // 4. AI 첫 인사 준비
const openingText = data?.ai_text || '';

// 기존 첫 인사 애니메이션이 실행 중이면 중지
if (openingTypingRef.current) {
  clearInterval(openingTypingRef.current);
  openingTypingRef.current = null;
}

// 첫 인사 글자 초기화
setOpeningDisplayText('');

// 첫 인사 말풍선 등록
setActiveDialogue(
  openingText
    ? [{
        id: `ai_opening_${Date.now()}`,
        speaker: 'AI',
        text: openingText,
        isOpening: true,
      }]
    : []
);

// 5. 대화 화면으로 이동
setStep('CHAT');

setAiStatus('THINKING');
setRecordingState('IDLE');

// 6. AI 첫 인사 애니메이션 및 Kokoro 음성 재생
if (openingText) {

  // 예상 음성 재생 시간 계산
  const estimatedDuration = (openingText.length / 9.5) * 1000;

  // 텍스트 애니메이션 실행
  animateOpeningText(openingText, estimatedDuration);

  try {
    // 백엔드에서 받은 Kokoro 음성 재생
    await playKokoroAudio(data?.audio_base64);

    if (openingTypingRef.current) {
      clearInterval(openingTypingRef.current);
      openingTypingRef.current = null;
    }

    setOpeningDisplayText(openingText);
    setAiStatus('IDLE');
    setRecordingState('IDLE');

  } catch (error) {
    console.error('❌ Kokoro 첫 인사 재생 오류:', error);

    if (openingTypingRef.current) {
      clearInterval(openingTypingRef.current);
      openingTypingRef.current = null;
    }

    setOpeningDisplayText(openingText);
    setAiStatus('IDLE');
    setRecordingState('IDLE');
  }

} else {
  setAiStatus('IDLE');
  setRecordingState('IDLE');
}

  } catch (error) {
    console.error('❌ 대화 시작 실패:', error);

    alert(
      error?.message ||
      'AI 대화를 시작하는 중 오류가 발생했습니다.'
    );

    setAiStatus('IDLE');
    setRecordingState('IDLE');
  }
};

const checkPendingClosing = async (sid, retryCount = 0) => {
  try {
    console.log(`🔎 마무리 답변 확인 ${retryCount + 1}회`);

    const response = await fetch(
      `${BASE_URL}/voice/conversation/${sid}/pending`,
      {
        method: 'GET',
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      }
    );

    const data = await response.json();

   console.log('🔚 pending 응답:', {
  ...data,
  response: data?.response
    ? {
        ...data.response,
        audio_base64: data.response?.audio_base64
          ? '[음성 데이터 생략]'
          : undefined,
      }
    : undefined,
});

    // 마무리 답변이 아직 생성되지 않은 경우
    if (!data?.pending) {
      if (retryCount < 10) {
        setTimeout(() => {
          checkPendingClosing(sid, retryCount + 1);
        }, 1000);
      } else {
        console.error('❌ 마무리 답변 확인 시간 초과');
        setAiStatus('ENDED');
        setRecordingState('ENDED');
      }

      return;
    }

    // 마무리 답변이 생성된 경우
    const closingText = data.response?.ai_text;
    const closingAudio = data.response?.audio_base64;

    if (!closingText) {
      console.error('❌ 마무리 답변 텍스트가 없습니다.');
      setAiStatus('ENDED');
      setRecordingState('ENDED');
      return;
    }

    console.log('🎉 마무리 답변 확인:', closingText);

    setAiStatus('THINKING');
    setRecordingState('IDLE');

setActiveDialogue(prev => [
  ...prev,
  {
    id: `ai_closing_${Date.now()}`,
    speaker: 'AI',
    text: closingText,
  },
]);


    // Kokoro 마무리 음성 재생
    try {
      console.log('🔊 Kokoro 마무리 음성 재생 시작');
      console.log('🎤 음성 데이터 존재 여부:', !!closingAudio);

      await playKokoroAudio(closingAudio);

      console.log('🔚 마무리 답변 재생 완료');

      // ==========================================
      // 백엔드에 재생 완료 알림
      // ==========================================
      try {
        const stateResponse = await fetch(
          `${BASE_URL}/voice/conversation/${sid}/state`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'ngrok-skip-browser-warning': 'true',
            },
            body: JSON.stringify({
              event: 'ai_playback_finished',
            }),
          }
        );

        const stateData = await stateResponse.json();

        console.log('📡 마무리 종료 상태:', stateData);

        setAiStatus('ENDED');
        setRecordingState('ENDED');

      } catch (error) {
        console.error('❌ 마무리 종료 상태 전송 실패:', error);

        setAiStatus('ENDED');
        setRecordingState('ENDED');
      }

    } catch (error) {
      console.error('❌ Kokoro 마무리 음성 재생 실패:', error);

      setAiStatus('ENDED');
      setRecordingState('ENDED');
    }


  } catch (error) {
    console.error(
      '❌ pending 마무리 답변 확인 실패:',
      error
    );

    if (retryCount < 10) {
      setTimeout(() => {
        checkPendingClosing(sid, retryCount + 1);
      }, 1000);
    } else {
      setAiStatus('ENDED');
      setRecordingState('ENDED');
    }
  }
};

const sendAudioToBackend = async (audioUri) => {
  try {
    console.log('📤 음성 대화 서버 전송:', audioUri);

    if (!sessionId) {
      throw new Error('대화 세션 ID가 없습니다.');
    }

    const formData = new FormData();

    formData.append('session_id', sessionId);

const audioFile = new File(audioUri);

formData.append('audio', audioFile);

    const response = await fetch(
      `${BASE_URL}/voice/conversation/message`,
      {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      }
    );

    console.log(
      '📡 음성 대화 응답 상태:',
      response.status
    );

const data = await response.json();

console.log('🤖 음성 대화 응답:', {
  success: data?.success,
  user_text: data?.user_text,
  ai_text: data?.ai_text,
  message: data?.message,
});
// 그 다음 마무리 처리
if (!response.ok || !data?.success) {
  if (data?.message === '마무리 답변 처리 중입니다.') {

    // 🔹 마지막 사용자 문장 먼저 화면에 추가
    if (data?.user_text) {
      setActiveDialogue(prev => [
        ...prev,
        {
          id: `user_${Date.now()}`,
          speaker: 'USER',
          text: data.user_text,
        },
      ]);
    }

    console.log(
      '🔚 백엔드에서 마무리 답변 처리 중 → pending 확인'
    );

    checkPendingClosing(sessionId);
    return;
  }

  throw new Error(
    data?.message || `HTTP ${response.status}`
  );
}
    // ==========================
    // 사용자 음성 인식 결과
    // ==========================

    const userText = data?.user_text || '';

    // ==========================
    // AI 답변
    // ==========================

    const aiText = data?.ai_text || '';

    console.log('🗣️ 사용자:', userText);
    console.log('🤖 AI:', aiText);

    // ==========================
    // 대화 내용 화면에 추가
    // ==========================

    const newMessages = [];

    if (userText) {
      newMessages.push({
        id: `user_${Date.now()}`,
        speaker: 'USER',
        text: userText,
      });
    }

    if (aiText) {
      newMessages.push({
        id: `ai_${Date.now() + 1}`,
        speaker: 'AI',
        text: aiText,
      });
    }

if (newMessages.length > 0) {
  setActiveDialogue(prev => [
    ...prev,
    ...newMessages,
  ]);
}

// ==========================
// AI 음성 재생
// ==========================


// ==========================
// AI Kokoro 음성 재생
// ==========================

if (aiText) {
  console.log('🔊 AI 답변 Kokoro 음성 재생');
  console.log('🔚 is_closing:', data?.is_closing);
  console.log('🎤 음성 데이터 존재 여부:', !!data?.audio_base64);

  try {
    // 백엔드에서 전달받은 Kokoro 음성 재생
    await playKokoroAudio(data?.audio_base64);

    console.log('🔊 AI 음성 재생 완료');

    // ==========================================
    // 🔚 마무리 답변이면 → 백엔드에 재생 완료 알림
    // ==========================================
    if (data?.is_closing) {
      console.log('🔚 마무리 멘트 재생 완료 → 대화 종료 요청');

      try {
        const stateResponse = await fetch(
          `${BASE_URL}/voice/conversation/${sessionId}/state`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'ngrok-skip-browser-warning': 'true',
            },
            body: JSON.stringify({
              event: 'ai_playback_finished',
            }),
          }
        );

        const stateData = await stateResponse.json();

        console.log('📡 대화 종료 응답:', stateData);

        if (stateData?.success) {
          console.log('✅ 대화가 정상적으로 종료되었습니다.');

          setAiStatus('ENDED');
          setRecordingState('ENDED');

          return;
        }

        console.warn('⚠️ 대화 종료 처리 실패:', stateData);

      } catch (error) {
        console.error('❌ 대화 종료 상태 전송 실패:', error);
      }

      // 종료 요청 실패 시에는 다시 녹음 가능하게
      setAiStatus('IDLE');
      setRecordingState('IDLE');

      return;
    }

    // ==========================================
    // 🔄 일반 AI 답변 → 다시 녹음 가능
    // ==========================================
    console.log('🔄 일반 AI 답변 완료 → 다시 녹음 가능');

    setRecordingState('IDLE');
    setAiStatus('IDLE');

  } catch (error) {
    console.error('❌ Kokoro AI 음성 재생 오류:', error);

    setRecordingState('IDLE');
    setAiStatus('IDLE');
  }
}


} catch (err) {
  console.error(
    '❌ 음성 대화 처리 에러:',
    err
  );

  // 🔚 이미 마무리 처리 중인 세션이면
  // 다시 녹음 가능 상태로 돌리지 않음
  if (
    err?.message === '마무리 답변 처리 중입니다.'
  ) {
    console.log(
      '🔚 백엔드에서 마무리 답변 처리 중이라고 응답했습니다.'
    );

    setAiStatus('ENDED');
    setRecordingState('ENDED');

    return;
  }

  setRecordingState('IDLE');
  setAiStatus('IDLE');

  alert(
    err?.message ||
    '음성 대화 처리 중 오류가 발생했습니다.'
  );
}

};  // ⭐ sendAudioToBackend 함수 닫기

const handleMicButtonPress = async () => {
  // AI가 생각 중이면 아무것도 하지 않음
  if (aiStatus === 'THINKING') {
    return;
  }

  // 🔚 대화가 완전히 종료된 경우
  if (
    aiStatus === 'ENDED' ||
    recordingState === 'ENDED'
  ) {
    console.log('🔚 이미 종료된 대화입니다.');
    return;
  }

  // 🎙️ 녹음 중이면 → 녹음 종료
  if (recordingState === 'RECORDING') {
    await stopRecording();
    return;
  }

  // 🟢 대기 중이면 → 녹음 시작
  if (recordingState === 'IDLE') {
    await startRecording();
  }
};


// 📌 기존 코드를 대체하는 새로운 State & 핸들러
const [flaggedMap, setFlaggedMap] = useState({});
const handleToggleFlag = (idx, chatItem) => {
  const itemKey = chatItem.id || `chat_${idx}`;

// onSaveToFeedback: 피드백 보관함 컴포넌트/상위 상태로 저장된 항목을 넘겨주는 함수 (선택)
setFlaggedMap((prev) => {
    const isCurrentlyFlagged = !!prev[itemKey]; // 👈 prev[itemKey]로 검사해야 정확합니다!
    const nextFlaggedState = !isCurrentlyFlagged;
    
    const updated = { ...prev };

    if (isCurrentlyFlagged) {
      delete updated[itemKey]; // 토글 OFF
    } else {
      updated[itemKey] = {     // 토글 ON
        id: itemKey,
        en: chatItem.text || chatItem.en || '',
        ko: chatItem.subText || chatItem.ko || '',
        topic: typeof currentTopic !== 'undefined' ? currentTopic : 'General',
        savedAt: new Date().toLocaleDateString(),
      };
    }

    // 💡 외부 콜백(onSaveToFeedback)이 있다면 상태 계산 직후 내부에서 안전하게 실행
    if (onSaveToFeedback) {
      const targetItem = updated[itemKey] || { id: itemKey };
      onSaveToFeedback(targetItem, nextFlaggedState);
    }

    return updated;
  });
};




// 피드백 보관함으로 이동하는 함수/버튼에서 실행
const handleGoToFeedback = () => {
  console.log('🎯 FeedbackScreen 이동 전 sessionId:', sessionId);

  if (!sessionId) {
    console.error('❌ sessionId가 없습니다!');
    return;
  }

  router.push({
    pathname: '/FeedbackScreen',
    params: {
      sessionId: String(sessionId),
    },
  });
};
  // 6. 유저 메시지 전송 및 AI 응답 처리
const sendUserMessageToBackend = async (userText) => {
  if (!userText?.trim()) {
    console.warn('⚠️ 전송할 사용자 문장이 없습니다.');
    return;
  }

  setActiveDialogue((prev) => [
    ...prev,
    {
      id: `user_${Date.now()}`,
      speaker: 'USER',
      text: userText,
      paraphrase:
        'I feel great to connect with you today as well!',
    },
  ]);

  setAiStatus('THINKING');

  if (USE_MOCK) {
    setTimeout(() => {
      setActiveDialogue((prev) => [
        ...prev,
        {
          id: `ai_${Date.now()}`,
          speaker: 'AI',
          text:
            'What kind of coffee do you usually enjoy in a cafe?',
          subText:
            '카페에서 보통 어떤 종류의 커피를 즐겨 마셔요?',
          suggestion: {
            en:
              'I usually like to drink an iced Americano.',
            ko:
              '저는 보통 아이스 아메리카노를 즐겨 마셔요.',
          },
        },
      ]);

      setAiStatus('IDLE');
      setRecordingState('IDLE');

      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({
          animated: true,
        });
      }, 100);

    }, 1200);

    return;
  }

  if (!sessionId) {
    console.error(
      '❌ sessionId가 없습니다. 대화를 먼저 시작해야 합니다.'
    );

    setAiStatus('IDLE');
    setRecordingState('IDLE');

    alert(
      '대화 세션이 없습니다. 주제를 다시 선택해주세요.'
    );

    return;
  }

  try {
    console.log(
      '📤 /conversation/message 전송:',
      {
        session_id: sessionId,
        user_message: userText,
      }
    );

    const response = await fetch(
      `${BASE_URL}/conversation/message`,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },

        body: JSON.stringify({
          session_id: sessionId,
          user_message: userText,
        }),
      }
    );

    const data = await response.json();

    console.log(
      '🤖 AI 응답:',
      JSON.stringify(data, null, 2)
    );

    if (!response.ok) {
      throw new Error(
        data?.message ||
        data?.error ||
        `HTTP ${response.status}`
      );
    }
    

if (data?.success && data?.ai_text) {

  console.log('🤖 AI:', data.ai_text);
  console.log('🎤 AI 음성 데이터 존재 여부:', !!data?.audio_base64);
console.log('🎤 AI 음성 형식:', data?.format);
console.log('🎤 AI 샘플레이트:', data?.sample_rate);

  // 화면에 AI 답변 추가
  setActiveDialogue((prev) => [
    ...prev,
    {
      id: `ai_${Date.now()}`,
      speaker: 'AI',
      text: data.ai_text,
      subText: '',
      suggestion: null,
    },
  ]);

 console.log('🔊 AI 답변 음성 재생');

  await Speech.stop();

  Speech.speak(data.ai_text, {
    language: 'en-US',
    rate: 0.95,
    pitch: 1.0,

    onDone: async () => {
  // ⭐ 모든 AI 답변은 동일하게 처리
  console.log('🔄 AI 답변 재생 완료 → 다시 녹음 가능');

  setRecordingState('IDLE');
  setAiStatus('IDLE');
},
});

} else {
  console.warn(
    '⚠️ AI 응답이 없습니다:',
    data
  );
}

  } catch (error) {
    console.error(
      '❌ /conversation/message 처리 에러:',
      error
    );

    setAiStatus('IDLE');
    setRecordingState('IDLE');

    alert(
      error?.message ||
      'AI 응답 처리 중 오류가 발생했습니다.'
    );
  }
};
const toggleParaphrase = (index) => {
  setShowParaphraseMap((prev) => ({
    ...prev,
    [index]: !prev[index],
  }));
};

if (!permission) return <View style={styles.center} />;

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <TouchableOpacity onPress={requestPermission} style={styles.permissionButton}>
          <Text style={styles.permissionText}>카메라 권한 허용</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* ---------------- 1. 카메라 스캐너 ---------------- */}
      {step === 'CAMERA' && (
        <CameraView style={styles.camera} facing="back" ref={cameraRef}>
          <View style={styles.overlay}>
           <View style={styles.header}>
  {/* 뒤로가기 버튼 */}
  <TouchableOpacity
    style={styles.backButton}
    onPress={() => router.back()}
    activeOpacity={0.7}
  >
    <Ionicons name="arrow-back" size={28} color="white" />
  </TouchableOpacity>

  <Text style={styles.headerTitle}>주변을 촬영해주세요!</Text>

  {/* 제목을 중앙에 유지하기 위한 빈 공간 */}
  <View style={{ width: 38 }} />
</View>

            <View style={styles.scanBanner}>
              <Text
  style={{
    color: 'white',
    marginTop: 8,
    fontSize: 13,
    fontWeight: '600',
  }}
>
  {captureDirection === 'front' && '1 / 4'}
  {captureDirection === 'right' && '2 / 4'}
  {captureDirection === 'back' && '3 / 4'}
  {captureDirection === 'left' && '4 / 4'}
</Text>
              <Text style={styles.scanText}>
  {USE_MOCK
    ? '🧪 Mock 테스트 모드'
    : CAPTURE_DIRECTIONS.find(
        (item) => item.key === captureDirection
      )?.instruction
  }
</Text>
            </View>
            <View style={styles.bottomControls}>
              <TouchableOpacity 
          style={styles.galleryIconButton} 
          onPress={handlePickImages}
          activeOpacity={0.8}
        >
          <Ionicons name="images-outline" size={26} color="white" />
        </TouchableOpacity>

              <TouchableOpacity style={styles.mainShutter} onPress={handleTakePicture} />
            </View>
          </View>
        </CameraView>
      )}

      {/* ---------------- 2. 분석 중 로딩 ---------------- */}
      {step === 'ANALYZING' && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#3b82f6" />
          <Text style={{ color: 'white', marginTop: 15, fontSize: 16, fontWeight: '600' }}>
            공간과 사물을 인식하고 있습니다...
          </Text>
        </View>
      )}

{/* ---------------- 3. 촬영 후 학습 화면 ---------------- */}
{step === 'MAIN_LEARNING' && capturedPhotoUris.length > 0 && (
  <View style={styles.backgroundImage}>
    {/* 촬영 후 화면 상단 헤더 */}
<View style={styles.learningHeader}>

  {/* 뒤로가기 */}
  <TouchableOpacity
    style={styles.learningHeaderBtn}
    onPress={() => router.back()}
    activeOpacity={0.7}
  >
    <Ionicons name="arrow-back" size={25} color="white" />
  </TouchableOpacity>

  {/* 오른쪽 재촬영 */}
  <TouchableOpacity
    style={styles.learningHeaderBtn}
    onPress={() => {
setCapturedPhotoUri(null);
setCapturedPhotoUris([]);
setCurrentPhotoIndex(0);
setRawViews(null);

  setCapturedPhotos({
    front: null,
    right: null,
    back: null,
    left: null,
  });

  setCaptureDirection('front');

  setSelectedObject(null);
  setDetectedObjects([]);
  setRecommendedTopics([]);

  setStep('CAMERA');
}}
    activeOpacity={0.7}
  >
    <Ionicons name="camera-reverse-outline" size={24} color="white" />
  </TouchableOpacity>

</View>
    <View style={styles.topCenterSpaceContainer}>
      <View style={styles.placeBadge}>
        <Text style={styles.placeText}>
  {currentSpace ? currentSpace.toUpperCase() : ''}
</Text>
      </View>
    </View>

{/* 사진 슬라이더 + 방향별 YOLO 바운딩 박스 */}
<FlatList
  style={StyleSheet.absoluteFill}
  onLayout={(event) => {
    const { width: w, height: h } = event.nativeEvent.layout;
    if (w > 0 && h > 0 && (w !== photoArea.width || h !== photoArea.height)) {
      setPhotoArea({ width: w, height: h });
    }
  }}
  data={capturedPhotoUris}
  horizontal
  pagingEnabled
  showsHorizontalScrollIndicator={false}
  keyExtractor={(item, index) => `${item}-${index}`}
  onMomentumScrollEnd={(event) => {
    const index = Math.round(event.nativeEvent.contentOffset.x / photoArea.width);
    setCurrentPhotoIndex(index);

    const directions = ['front', 'right', 'back', 'left'];
    const currentDirection = directions[index];

    setDetectedObjects(directionObjects[currentDirection] || []);
    setSelectedObject(null);
  }}
  renderItem={({ item, index }) => {
    const directions = ['front', 'right', 'back', 'left'];
    const direction = directions[index];
    const objects = directionObjects[direction] || [];

    return (
      <View
        style={{
          width: photoArea.width,
          height: '100%',
          position: 'relative',
        }}
      >
        {/* 현재 방향 사진 */}
        <ImageBackground
          source={{ uri: item }}
          style={{
            width: '100%',
            height: '100%',
          }}
          resizeMode="cover"
        />


{/* 현재 사진의 객체 도트 및 말풍선 */}
{objects.map((obj, objectIndex) => {
  const x = Number(obj?.x);
  const y = Number(obj?.y);

  if (
    !Number.isFinite(x) ||
    !Number.isFinite(y)
  ) {
    return null;
  }

  const isSelected = selectedObject?.id === obj?.id;

  return (
    <React.Fragment key={obj?.id ?? `${direction}_${objectIndex}`}>
      {/* 객체 도트 */}
      <TouchableOpacity
        style={[
          styles.objectDot,
          {
            left: x - 7,
            top: y - 7,
          },
          isSelected && styles.selectedObjectDot,
        ]}
        onPress={() => {
          setSelectedObject((prev) =>
            prev?.id === obj?.id ? null : obj
          );
        }}
        activeOpacity={0.8}
      />

      {/* 선택된 객체의 말풍선 */}
      {isSelected && (
        <View
          style={[
            styles.objectTooltip,
            {
              left: Math.max(
                8,
                Math.min(x + 12, photoArea.width - 170)
              ),
              top: Math.max(
                8,
                Math.min(y - 25, photoArea.height - 65)
              ),
            },
          ]}
        >
          <View style={styles.objectTooltipContent}>
            <Text style={styles.objectEnglishText}>
              {obj?.name}
            </Text>
          </View>

          <TouchableOpacity
  style={styles.objectSpeakerButton}
  onPress={async () => {
    const savedVoice = await AsyncStorage.getItem('voiceIdentifier');

    Speech.stop();

    Speech.speak(obj?.name || 'Object', {
      language: 'en-US',
      voice: savedVoice || undefined,
      rate: 0.85,
      pitch: 1.0,
    });
  }}
  activeOpacity={0.7}
>
            <Ionicons
              name="volume-high"
              size={19}
              color="#3B82F6"
            />
          </TouchableOpacity>
        </View>
      )}
    </React.Fragment>
  );
})}

      </View>
    );
  }}
/>

    {/* AI 추천 주제 */}
{!selectedObject && (
  <View style={styles.topicPreviewContainer}>

    {/* 접혀 있는 상태에서도 항상 보이는 헤더 */}
    <TouchableOpacity
      style={styles.topicPreviewHeader}
      onPress={() =>
        setShowRecommendedTopics((prev) => !prev)
      }
      activeOpacity={0.8}
    >
      <View style={styles.topicPreviewTitleRow}>
        <Ionicons
          name="sparkles"
          size={18}
          color="#3B82F6"
        />

        <Text style={styles.topicPreviewTitle}>
          AI 추천 주제
        </Text>
      </View>

      <View style={styles.topicHeaderRight}>
        <Text style={styles.topicViewAllText}>
          {showRecommendedTopics ? '접기' : '펼치기'}
        </Text>

        <Ionicons
          name={
            showRecommendedTopics
              ? "chevron-down"
              : "chevron-up"
          }
          size={17}
          color="#64748B"
        />
      </View>
    </TouchableOpacity>

    {/* 펼쳤을 때만 추천 주제 표시 */}
    {showRecommendedTopics && (
      <View style={styles.topicPreviewList}>

      
{recommendedTopics.slice(0, 2).map((item, index) => (
  <TouchableOpacity
    key={item.topic_id || index}
    style={styles.topicPreviewItem}
    onPress={() => startConversationAPI(item)}
    activeOpacity={0.8}
  >
    <View style={styles.topicPreviewIcon}>
      <Ionicons
        name={
          index === 0
            ? "cafe-outline"
            : "chatbubble-ellipses-outline"
        }
        size={18}
        color="#3B82F6"
      />
    </View>

    <Text
      style={styles.topicPreviewText}
      numberOfLines={1}
    >
      {item.title_ko || item.title_en}
    </Text>

    <Ionicons
      name="chevron-forward"
      size={17}
      color="#94A3B8"
    />
  </TouchableOpacity>
))}


{/* 자유주제 대화 */}
<TouchableOpacity
  style={styles.topicPreviewItem}
  onPress={startFreeConversationAPI}
  activeOpacity={0.8}
>
  <View style={styles.topicPreviewIcon}>
    <Ionicons
      name="chatbubbles-outline"
      size={18}
      color="#3B82F6"
    />
  </View>

  <Text style={styles.topicPreviewText}>
    자유주제 대화
  </Text>

  <Ionicons
    name="chevron-forward"
    size={17}
    color="#94A3B8"
  />
</TouchableOpacity>


        {/* 전체 보기 */}
        <TouchableOpacity
          style={styles.topicViewAllBtn}
          onPress={() => setStep('TOPIC_ROADMAP')}
          activeOpacity={0.7}
        >
          <Text style={styles.topicViewAllText}>
            전체 보기
          </Text>

          <Ionicons
            name="chevron-forward"
            size={16}
            color="#64748B"
          />
        </TouchableOpacity>

      </View>
    )}

  </View>
)}
  </View>
)}

{/* ---------------- 4. 로드맵 전체 페이지 (일자 타임라인) ---------------- */}
{step === 'TOPIC_ROADMAP' && (
  <View style={styles.roadmapWarmContainer}>
    {/* 상단 네비게이션 헤더 */}
    <View style={styles.roadmapWarmHeader}>
      <TouchableOpacity
        style={styles.roadmapWarmBackBtn}
        onPress={() => setStep('MAIN_LEARNING')}
        activeOpacity={0.7}
      >
        <Ionicons name="chevron-back" size={24} color="#1E293B" />
      </TouchableOpacity>

      <Text style={styles.roadmapWarmTitle}>AI 맞춤 학습 코스</Text>

      <View style={styles.warmHeaderBadge}>
        <Ionicons name="sparkles" size={13} color="#3B82F6" />
        <Text style={styles.warmHeaderBadgeText}>{recommendedTopics.length} Steps</Text>
      </View>
    </View>

    <ScrollView
      showsVerticalScrollIndicator={false}
      contentContainerStyle={styles.roadmapWarmScrollContent}
    >
      {/* 히어로 공간 카드 */}
      <View style={styles.warmSpaceHeroCard}>
        <View style={styles.warmSpaceHeroTag}>
          <Ionicons name="location" size={13} color="#3B82F6" />
          <Text style={styles.warmSpaceHeroTagText}>LOCATION DETECTED</Text>
        </View>
        <Text style={styles.warmSpaceHeroTitle}>
          [ {currentSpace ? currentSpace.toUpperCase() : ''} ]
        </Text>
        <Text style={styles.warmSpaceHeroSub}>
          상황별 핵심 표현을 단계별로 익혀보세요
        </Text>
      </View>

      {/* 자유주제 대화 버튼 */}
      <TouchableOpacity
        style={styles.warmFreeTopicButton}
        onPress={startFreeConversationAPI}
        activeOpacity={0.8}
      >
        <Ionicons name="chatbubbles-outline" size={22} color="#070e72" />

        <View style={{ flex: 1 }}>
          <Text style={styles.warmFreeTopicBtnTitle}>
            자유주제로 대화하기
          </Text>
          <Text style={styles.warmFreeTopicBtnSub}>
            원하는 주제로 영어 회화를 시작해 보세요.
          </Text>
        </View>

        <Ionicons name="chevron-forward" size={20} color="#070e72" />
      </TouchableOpacity>

      {/* 깔끔한 일자 타임라인 로드맵 */}
      <View style={styles.cleanRoadmapList}>
        {/* 왼쪽에 들어가는 수직 연결선 */}
        <View style={styles.cleanVerticalLine} />

        {recommendedTopics.map((item, index) => {
          const isFirst = index === 0;
          const topicText =
            typeof item === 'object'
              ? item.title_ko || item.title_en || item.topic || ''
              : item;

          return (
            <TouchableOpacity
              key={index}
              style={[
                styles.cleanRoadmapNode,
                isFirst && styles.activeCleanNode,
              ]}
              onPress={() => startConversationAPI(item)}
              activeOpacity={0.85}
            >
              {/* 좌측 스텝 번호 원형 배지 */}
              <View
                style={[
                  styles.cleanNodeBadge,
                  isFirst && styles.activeCleanNodeBadge,
                ]}
              >
                <Text
                  style={[
                    styles.cleanNodeBadgeText,
                    isFirst && styles.activeCleanNodeBadgeText,
                  ]}
                >
                  {index + 1}
                </Text>
              </View>

              {/* 중앙 주제 텍스트 & START 태그 */}
              <View style={styles.cleanNodeTextWrapper}>
                <Text
                  style={[
                    styles.cleanNodeText,
                    isFirst && styles.activeCleanNodeText,
                  ]}
                  numberOfLines={1}
                >
                  {topicText}
                </Text>

                {isFirst && (
                  <View style={styles.warmStartBadge}>
                    <Text style={styles.warmStartBadgeText}>START</Text>
                  </View>
                )}
              </View>

              <Ionicons
                name={isFirst ? "play-circle" : "chevron-forward"}
                size={isFirst ? 22 : 18}
                color={isFirst ? "#3B82F6" : "#94A3B8"}
              />
            </TouchableOpacity>
          );
        })}
      </View>
    </ScrollView>
  </View>
)}

{/* ---------------- 5. 대화 화면 (타이머 & 피드백 버튼 복구) ---------------- */}
{step === 'CHAT' && (
  <View style={styles.chatWarmContainer}>
    {/* 상단 헤더: 뒤로가기 / 주제 / 타이머 */}
    <View style={styles.chatWarmHeader}>
      <TouchableOpacity
        onPress={() => setStep('MAIN_LEARNING')}
        style={styles.chatWarmBackBtn}
        activeOpacity={0.7}
      >
        <Ionicons name="chevron-back" size={24} color="#1E293B" />
      </TouchableOpacity>

      {/* 선택한 추천 주제 (중앙) */}
      <View style={{ alignItems: 'center', flex: 1, paddingHorizontal: 8 }}>
        <Text style={styles.chatWarmHeaderTitle} numberOfLines={1}>
          {currentTopic || '자유 대화'}
        </Text>
      </View>

      {/* 우측 타이머 + AI 응답 표시 */}
<View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
  <View style={styles.timerBadgeWarm}>
    <Ionicons
      name="time-outline"
      size={13}
      color="#64748B"
      style={{ marginRight: 3 }}
    />
    <Text style={styles.timerTextWarm}>
      {formatTime(timeLeft)}
    </Text>
  </View>

  {aiStatus === 'THINKING' && (
    <View
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
        width: 38,
        height: 32,
        backgroundColor: '#334155',
        borderRadius: 16,
      }}
    >
      {thinkingDots.map((dot, index) => (
        <Animated.View
          key={index}
          style={{
            width: 5,
            height: 5,
            borderRadius: 3,
            backgroundColor: dot.interpolate({
              inputRange: [0, 1],
              outputRange: ['#858585', '#FFFFFF'],
            }),
          }}
        />
      ))}
    </View>
  )}
</View>
    </View>

    {/* 대화 스크롤 영역 */}
    <ScrollView
      ref={scrollViewRef}
      style={styles.chatWarmScrollView}
      contentContainerStyle={{ paddingBottom: 20 }}
      showsVerticalScrollIndicator={false}
    >
      {activeDialogue.map((chat, idx) => {
        const isAI = chat.speaker === 'AI';

        return (
          <View key={idx} style={{ marginBottom: 14 }}>
            {!isAI ? (
              /* 유저 말풍선 (우측, 연파랑) */
              <View style={{ alignItems: 'flex-end', marginLeft: 'auto', maxWidth: '82%' }}>
                <View style={styles.userBubbleWarm}>
  {chat.isUserThinking ? (
    <View
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        gap: 5,
        paddingVertical: 4,
      }}
    >
      {thinkingDots.map((dot, index) => (
        <Animated.View
          key={index}
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            backgroundColor: dot.interpolate({
              inputRange: [0, 1],
              outputRange: ['#B8B8B8', '#2563EB'],
            }),
          }}
        />
      ))}
    </View>
  ) : (
    <Text style={styles.userBubbleTextWarm}>
      {chat.text}
    </Text>
  )}
</View>
              </View>
            ) : (
              /* AI 말풍선 (좌측, 흰색) */
              <View style={{ alignItems: 'flex-start', maxWidth: '82%' }}>
 <View style={styles.aiBubbleWarm}>
  {(recordingState === 'RECORDING' || aiStatus === 'THINKING') && !chat.isOpening ? (
    <View style={{
      flexDirection: 'row',
      alignItems: 'center',
      gap: 5,
      paddingVertical: 4,
    }}>
      {thinkingDots.map((dot, index) => (
        <Animated.View
          key={index}
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            backgroundColor: dot.interpolate({
              inputRange: [0, 1],
              outputRange: ['#B8B8B8', '#64748B'],
            }),
          }}
        />
      ))}
    </View>
  ) : (
    <Text style={styles.aiBubbleTextWarm}>
      {chat.isOpening
        ? openingDisplayText
        : chat.text}
    </Text>
  )}
</View>
              </View>
            )}
          </View>
        );
      })}

    </ScrollView>

    {/* 하단 마이크 & 음파 & 피드백 버튼 통합 플레이트 */}
    <View style={styles.warmBottomArea}>
      <View style={styles.micWavePlate}>
        {/* 좌측 음파 애니메이션 */}
        <View style={styles.warmWaveformContainer}>
          {waveAnimValues.slice(0, 4).map((animVal, i) => (
            <Animated.View
              key={`left_${i}`}
              style={[
                styles.warmWaveBar,
                { height: animVal },
              ]}
            />
          ))}
        </View>

        {/* 중앙 원형 마이크 버튼 */}
        <TouchableOpacity
          style={styles.warmMicCircle}
          onPress={handleMicButtonPress}
          activeOpacity={0.8}
        >
          <Ionicons name="mic" size={26} color="#2563EB" />
        </TouchableOpacity>

        {/* 우측 피드백 이동 버튼 (조건부 노출) */}
        <View style={styles.warmSideActionRow}>
          {showFeedbackBtn ? (
            <TouchableOpacity
              style={styles.warmSideCircleBtn}
              onPress={handleGoToFeedback}
              activeOpacity={0.8}
            >
              <Ionicons name="happy-outline" size={22} color="#1E293B" />
            </TouchableOpacity>
          ) : (
            <View style={{ width: 38, height: 38 }} />
          )}
        </View>
      </View>
    </View>
  </View>
)}
    </View>
  );
}


const styles = StyleSheet.create({
  /* ================= 1. CAMERA 스캔 화면 ================= */
  container: { flex: 1, backgroundColor: '#000000' },
  camera: { flex: 1 },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', justifyContent: 'space-between' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingTop: 60, paddingHorizontal: 20 },
  headerTitle: { color: 'white', fontSize: 18, fontWeight: 'bold' },
  backButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanBanner: { alignItems: 'center', marginTop: '20%' },
  scanText: { color: 'white', fontSize: 15, fontWeight: '600', backgroundColor: 'rgba(0,0,0,0.7)', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 20 },
  bottomControls: { 
    flexDirection: 'row', 
    justifyContent: 'center', 
    alignItems: 'center', 
    marginBottom: 50,
    position: 'relative',
    width: '100%',
  },
  mainShutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: 'white', borderWidth: 5, borderColor: 'rgba(255,255,255,0.4)' },
  galleryIconButton: {
    position: 'absolute',   
    left: 40,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.45)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000' },
  permissionButton: { backgroundColor: '#3b82f6', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10 },
  permissionText: { color: 'white', fontWeight: 'bold' },

  /* ================= 2. MAIN_LEARNING (촬영 후 학습 & 바운딩 박스) ================= */
  backgroundImage: { flex: 1, resizeMode: 'cover' },
  retakeBtn: {
    position: 'absolute',
    top: 50,
    right: 20,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(15, 23, 42, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 50,
  },
  learningHeader: {
    position: 'absolute',
    top: 52,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 100,
  },
  learningHeaderBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255, 255, 255, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.9)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  topCenterSpaceContainer: { 
    marginTop: 60, 
    alignItems: 'center', 
    width: '100%',
    zIndex: 10,
  },
  placeBadge: { 
    backgroundColor: 'rgba(255, 255, 255, 0.88)', 
    paddingHorizontal: 28, 
    paddingVertical: 12, 
    borderRadius: 24, 
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    borderWidth: 1.5,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  placeText: { 
    fontSize: 24, 
    fontWeight: '900', 
    color: '#0F172A', 
    letterSpacing: 2,
  },

 // ==========================================
// 객체 도트 UI
// ==========================================

// 기본 흰색 도트
objectDot: {
  position: 'absolute',
  width: 14,
  height: 14,
  borderRadius: 7,
  backgroundColor: '#FFFFFF',
  borderWidth: 1.5,
  borderColor: '#CBD5E1',
  shadowColor: '#000',
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.25,
  shadowRadius: 3,
  elevation: 5,
  zIndex: 100,
},

selectedObjectDot: {
  backgroundColor: '#FFFFFF',
  borderColor: '#3B82F6',
  borderWidth: 2,
},

// 객체 정보 말풍선
objectTooltip: {
  position: 'absolute',
  flexDirection: 'row',
  alignItems: 'center',
  backgroundColor: '#FFFFFF',
  paddingHorizontal: 14,
  paddingVertical: 11,
  borderRadius: 14,
  borderWidth: 1,
  borderColor: '#E2E8F0',
  shadowColor: '#000',
  shadowOffset: { width: 0, height: 3 },
  shadowOpacity: 0.15,
  shadowRadius: 6,
  elevation: 10,
  zIndex: 200,
},

objectTooltipContent: {
  flexDirection: 'column',
  marginRight: 12,
},

objectEnglishText: {
  fontSize: 15,
  fontWeight: '800',
  color: '#1E293B',
},

objectKoreanText: {
  fontSize: 12,
  fontWeight: '500',
  color: '#64748B',
  marginTop: 3,
},

objectSpeakerButton: {
  width: 32,
  height: 32,
  borderRadius: 16,
  backgroundColor: '#EFF6FF',
  justifyContent: 'center',
  alignItems: 'center',
},

  /* 단어 팝업 카드 & AI 추천 주제 카셀 */
  wordPopupCard: {
    position: 'absolute',
    bottom: 105,
    left: '8%',
    right: '8%',
    height: 105,
    backgroundColor: 'rgba(255, 255, 255, 0.88)',
    borderRadius: 20,
    paddingHorizontal: 22,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.18,
    shadowRadius: 8,
    elevation: 8,
    zIndex: 100,
  },
  wordSubTitle: {
    fontSize: 11,
    color: '#64748B',
    fontWeight: '600',
    marginBottom: 4,
  },
  wordText: {
    fontSize: 24,
    fontWeight: '800',
    color: '#0F172A',
  },
  wordPopupActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  topicPreviewContainer: {
    position: 'absolute',
    bottom: 42,
    left: '7%',
    right: '7%',
    backgroundColor: 'rgba(255, 255, 255, 0.96)',
    borderRadius: 22,
    paddingHorizontal: 18,
    paddingVertical: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
    zIndex: 20,
  },
  topicPreviewHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  topicPreviewTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  topicPreviewTitle: {
    marginLeft: 7,
    color: '#0F172A',
    fontSize: 16,
    fontWeight: '800',
  },
  topicHeaderRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  topicViewAllBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
  },
  topicViewAllText: {
    color: '#64748B',
    fontSize: 12,
    fontWeight: '600',
  },
  topicPreviewList: {
    gap: 8,
  },
  topicPreviewItem: {
    height: 48,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  topicPreviewIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#EFF6FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  topicPreviewText: {
    flex: 1,
    color: '#1E293B',
    fontSize: 13,
    fontWeight: '600',
  },

  /* ================= 3. TOPIC_ROADMAP (일자 타임라인) ================= */
  roadmapWarmContainer: {
    flex: 1,
    backgroundColor: '#FAF7F2',
    paddingTop: 50,
  },
  roadmapWarmHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  roadmapWarmBackBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#EFECE6',
  },
  roadmapWarmTitle: {
    color: '#1E293B',
    fontSize: 18,
    fontWeight: '800',
  },
  warmHeaderBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#EFF6FF',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#DBEAFE',
  },
  warmHeaderBadgeText: {
    color: '#3B82F6',
    fontSize: 11,
    fontWeight: '700',
  },
  roadmapWarmScrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 80,
  },
  warmSpaceHeroCard: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    borderRadius: 24,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#EFECE6',
    shadowColor: '#1E293B',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 10,
    elevation: 2,
  },
  warmSpaceHeroTag: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginBottom: 4,
  },
  warmSpaceHeroTagText: {
    color: '#3B82F6',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  warmSpaceHeroTitle: {
    color: '#1E293B',
    fontSize: 22,
    fontWeight: '900',
  },
  warmSpaceHeroSub: {
    color: '#64748B',
    fontSize: 12,
    marginTop: 4,
  },
  warmFreeTopicButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#bad4e9',
    padding: 16,
    borderRadius: 20,
    marginBottom: 24,
    gap: 12,
    shadowColor: '#6C5CE7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },
  warmFreeTopicBtnTitle: {
    color: '#0f0505',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 2,
  },
  warmFreeTopicBtnSub: {
    color: 'rgba(49, 47, 47, 0.8)',
    fontSize: 11,
    lineHeight: 15,
  },
  cleanRoadmapList: {
    position: 'relative',
    width: '100%',
    paddingLeft: 4,
    gap: 12,
  },
  cleanVerticalLine: {
    position: 'absolute',
    top: 24,
    bottom: 24,
    left: 23,
    width: 2,
    backgroundColor: '#E2E8F0',
    zIndex: 1,
  },
  cleanRoadmapNode: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 20,
    width: '100%',
    borderWidth: 1,
    borderColor: '#ECEAE2',
    shadowColor: '#1E293B',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 6,
    elevation: 2,
    zIndex: 2,
  },
  activeCleanNode: {
    borderColor: '#3B82F6',
    borderWidth: 1.5,
    backgroundColor: '#FFFFFF',
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  cleanNodeBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#F1F5F9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },
  activeCleanNodeBadge: {
    backgroundColor: '#3B82F6',
  },
  cleanNodeBadgeText: {
    color: '#64748B',
    fontSize: 12,
    fontWeight: '800',
  },
  activeCleanNodeBadgeText: {
    color: '#FFFFFF',
  },
  cleanNodeTextWrapper: {
    flex: 1,
    justifyContent: 'center',
  },
  cleanNodeText: {
    color: '#334155',
    fontSize: 14,
    fontWeight: '600',
  },
  activeCleanNodeText: {
    color: '#0F172A',
    fontWeight: '800',
  },
  warmStartBadge: {
    alignSelf: 'flex-start',
    backgroundColor: '#3B82F6',
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 4,
    marginTop: 3,
  },
  warmStartBadgeText: {
    color: '#FFFFFF',
    fontSize: 8,
    fontWeight: '900',
  },

  //* --- 5단계 대화 화면 전용 스타일 --- */
  chatWarmContainer: {
    flex: 1,
    backgroundColor: '#FAF7F2',
    paddingTop: 50,
  },
  chatWarmHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 12,
  },
  chatWarmBackBtn: {
    width: 38,
    height: 38,
    justifyContent: 'center',
    alignItems: 'center',
  },
  chatWarmHeaderTitle: {
    color: '#1E293B',
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
  },
  timerBadgeWarm: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#EFECE6',
  },
  timerTextWarm: {
    color: '#64748B',
    fontSize: 12,
    fontWeight: '700',
  },
  chatWarmScrollView: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 12,
  },
  aiBubbleWarm: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderRadius: 22,
    borderBottomLeftRadius: 6,
    shadowColor: '#1E293B',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.03,
    shadowRadius: 6,
    elevation: 1,
  },
  aiBubbleTextWarm: {
    color: '#0F172A',
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '500',
  },
  userBubbleWarm: {
    backgroundColor: '#DCE6FF',
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderRadius: 22,
    borderBottomRightRadius: 6,
  },
  userBubbleTextWarm: {
    color: '#0F172A',
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '500',
  },
  warmBottomArea: {
    paddingBottom: 36,
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  micWavePlate: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FFFFFF',
    borderRadius: 32,
    paddingHorizontal: 20,
    paddingVertical: 12,
    shadowColor: '#1E293B',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 10,
    elevation: 2,
  },
  warmWaveformContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    width: 45,
    justifyContent: 'center',
  },
  warmWaveBar: {
    width: 3,
    backgroundColor: '#93C5FD',
    borderRadius: 2,
  },
  warmMicCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#DBEAFE',
    justifyContent: 'center',
    alignItems: 'center',
  },
  warmSideActionRow: {
    width: 45,
    alignItems: 'flex-end',
  },
  warmSideCircleBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#F8FAFC',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
});
