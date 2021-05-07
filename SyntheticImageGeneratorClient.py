import socket
import struct
from typing import List


class Client:
    def __init__(self, address='127.0.0.1', port=12583, unix=False):
        self.address = address
        self.port = port
        if unix:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect('/var/run/synthetic-image-generator.sock')
        else:
            self.socket = socket.socket()
            self.socket.connect((address, port))
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
            self.socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def __del__(self):
        self.socket.close()

    def __send__(self, bytesToSend: bytes):
        self.socket.send(_ByteConverter.toBytes(
            len(bytesToSend), "Int32") + bytesToSend)

    def __receive__(self):
        received_bytes = self.__receive_all_bytes__()
        errorCode, offset = _ByteConverter.fromBytes(
            received_bytes, 0, 'Int32')
        if errorCode == 0:
            return received_bytes, offset
        else:
            message, offset = _ByteConverter.fromBytes(
                received_bytes, offset, 'String')
            raise SyntheticImageGeneratorClientException(errorCode, message)

    def __receive_all_bytes__(self):
        raw_length = self.__receive_n_bytes__(4)
        if not raw_length:
            return None
        length, offset = _ByteConverter.fromBytes(raw_length, 0, 'Int32')
        return self.__receive_n_bytes__(length)

    def __receive_n_bytes__(self, length):
        data = bytearray()
        while len(data) < length:
            part = self.socket.recv(length - len(data))
            if not part:
                return None
            else:
                data.extend(part)
        return data

    def __enter__(self):
        self.BeginSession()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.EndSession()

    def sub(self, strings: List[str]) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.sub', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(strings, 'String[]')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def testMethodException(self, a: int, b: int) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.testMethodException', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(a, 'Int32')
        bytesToSend += _ByteConverter.toBytes(b, 'Int32')
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Int32')
        return result

    def testMethodAPlusB(self, a: int, b: int) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.testMethodAPlusB', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(a, 'Int32')
        bytesToSend += _ByteConverter.toBytes(b, 'Int32')
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Int32')
        return result

    def testMethodLotOfParamsReturnNothing(self, a: bool, b: str, b2: int, c: int, d: int, e: int, f: float, g: float, h: str, i: bytes) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.testMethodLotOfParamsReturnNothing', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(a, 'Boolean')
        bytesToSend += _ByteConverter.toBytes(b, 'Char')
        bytesToSend += _ByteConverter.toBytes(b2, 'Byte')
        bytesToSend += _ByteConverter.toBytes(c, 'Int16')
        bytesToSend += _ByteConverter.toBytes(d, 'Int32')
        bytesToSend += _ByteConverter.toBytes(e, 'Int64')
        bytesToSend += _ByteConverter.toBytes(f, 'Single')
        bytesToSend += _ByteConverter.toBytes(g, 'Double')
        bytesToSend += _ByteConverter.toBytes(h, 'String')
        bytesToSend += _ByteConverter.toBytes(i, 'Byte[]')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def getBooleanTrue(self) -> bool:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getBooleanTrue', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Boolean')
        return result

    def getBooleanFalse(self) -> bool:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getBooleanFalse', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Boolean')
        return result

    def getChar(self) -> str:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getChar', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Char')
        return result

    def getByte(self) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getByte', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Byte')
        return result

    def getShort(self) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getShort', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Int16')
        return result

    def getInt(self) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getInt', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Int32')
        return result

    def getLong(self) -> int:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getLong', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Int64')
        return result

    def getFloat(self) -> float:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getFloat', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Single')
        return result

    def getDouble(self) -> float:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getDouble', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Double')
        return result

    def getString(self) -> str:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getString', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'String')
        return result

    def getByteArra(self) -> bytes:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('RpcTest.getByteArra', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Byte[]')
        return result

    def BeginSession(self) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.BeginSession', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def EndSession(self) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.EndSession', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def GetRenderedImages(self) -> bytes:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.GetRenderedImages', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'Byte[]')
        return result

    def RenderImagesDontWait(self, images: List[List[bytes]]) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.RenderImagesDontWait', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(images, 'Byte[][][]')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def AugmentImages(self, images: List[List[bytes]], id: str) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.AugmentImages', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(images, 'Byte[][][]')
        bytesToSend += _ByteConverter.toBytes(id, 'String')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def GetClasses(self) -> List[str]:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.GetClasses', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        data, offset = self.__receive__()
        result, offset = _ByteConverter.fromBytes(data, offset, 'String[]')
        return result

    def CheckClasses(self, classes: List[str]) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.CheckClasses', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(classes, 'String[]')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def SetRandomSeed(self, seed: int) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.SetRandomSeed', 'String')
        # Parameters
        bytesToSend += _ByteConverter.toBytes(seed, 'Int32')
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def SetRandomStrategyToUniformDistribution(self) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.SetRandomStrategyToUniformDistribution', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def SetRandomStrategyToNormalDistribution(self) -> None:
        bytesToSend = bytes()
        # Message Code
        bytesToSend += _ByteConverter.toBytes('Session.SetRandomStrategyToNormalDistribution', 'String')
        # Parameters
        # Sending
        self.__send__(bytesToSend)
        # Result
        self.__receive__()

    def LogoClass(self, name: str) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoClass', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(name, 'String')
        # Finish
        return bytesOfTransformation

    def LogoRotationXY(self, tiltDegree: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoRotationXY', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(tiltDegree, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoRotationZ(self, degree: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoRotationZ', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(degree, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoScale(self, x: float, y: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoScale', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(x, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(y, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoScaleTrigonometric(self, angle: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoScaleTrigonometric', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(angle, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoScaleUniform(self, value: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoScaleUniform', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(value, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoMirrorHorizontal(self) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoMirrorHorizontal', 'String')
        # Parameters
        # Finish
        return bytesOfTransformation

    def LogoMirrorVertical(self) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoMirrorVertical', 'String')
        # Parameters
        # Finish
        return bytesOfTransformation

    def LogoTranslation(self, x: float, y: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoTranslation', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(x, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(y, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoLookAtCamera(self) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('Logo.LogoLookAtCamera', 'String')
        # Parameters
        # Finish
        return bytesOfTransformation

    def LogoMeshSubdivide(self, resolutionX: int, resolutionY: int) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('LogoMeshTransformer.LogoMeshSubdivide', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(resolutionX, 'Int32')
        bytesOfTransformation += _ByteConverter.toBytes(resolutionY, 'Int32')
        # Finish
        return bytesOfTransformation

    def LogoMeshPerlinNoise(self, prelinNoiseFrequencyX: float, prelinNoiseFrequencyY: float, perlinNoiseAmplitude: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('LogoMeshTransformer.LogoMeshPerlinNoise', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(prelinNoiseFrequencyX, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(prelinNoiseFrequencyY, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(perlinNoiseAmplitude, 'Single')
        # Finish
        return bytesOfTransformation

    def LogoMeshCurve(self, curveX: float, curveY: float, curveCenterX: float, curveCenterY: float, curveExponentX: float, curveExponentY: float) -> bytes:
        bytesOfTransformation = bytes()
        # Transformation Code
        bytesOfTransformation += _ByteConverter.toBytes('LogoMeshTransformer.LogoMeshCurve', 'String')
        # Parameters
        bytesOfTransformation += _ByteConverter.toBytes(curveX, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(curveY, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(curveCenterX, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(curveCenterY, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(curveExponentX, 'Single')
        bytesOfTransformation += _ByteConverter.toBytes(curveExponentY, 'Single')
        # Finish
        return bytesOfTransformation


class SyntheticImageGeneratorClientException(Exception):
    def __init__(self, errorCode: int, message: str):
        self.errorCode = errorCode
        self.message = message

    def __str__(self):
        return f'Error code: {self.errorCode} Message: {self.message}'


class _ByteConverter:
    @classmethod
    def fromBytes(cls, data: bytes, offset: int, targetType: str):
        if targetType.endswith('[]'):
            elementType = targetType[:-2]
            length, offset = cls.fromBytes(data, offset, 'Int32')

            if elementType == "Byte":
                return data[offset:offset + length], offset + length

            result = list()
            for i in range(0, length):
                element, offset = cls.fromBytes(data, offset, elementType)
                result.append(element)
            return result, offset
        else:
            if targetType == 'Boolean':
                return struct.unpack_from('?', data, offset)[0], offset+1
            if targetType == 'Char':
                return data[offset:offset+2].decode('utf-16-le'), offset+2
            if targetType == 'Byte':
                return data[offset], offset+1
            if targetType == 'Int16':
                return struct.unpack_from('h', data, offset)[0], offset+2
            if targetType == 'Int32':
                return struct.unpack_from('i', data, offset)[0], offset+4
            if targetType == 'Int64':
                return struct.unpack_from('q', data, offset)[0], offset+8
            if targetType == 'Single':
                return struct.unpack_from('f', data, offset)[0], offset+4
            if targetType == 'Double':
                return struct.unpack_from('d', data, offset)[0], offset+8
            if targetType == 'String':
                length, offset = cls.fromBytes(data, offset, 'Int32')
                rawMessage = struct.unpack_from(
                    '%ds' % length, data, offset)[0]
                message = rawMessage.decode('utf-16-le')
                return message, offset + length

    @classmethod
    def toBytes(cls, obj, targetType: str):
        if targetType.endswith('[]'):
            elementType = targetType[:-2]
            length = len(obj)
            bytesToWrite = cls.toBytes(length, 'Int32')
            for i in range(0, length):
                bytesToWrite += cls.toBytes(obj[i], elementType)
            return bytesToWrite
        else:
            if targetType == 'Boolean':
                return struct.pack('?', obj)
            if targetType == 'Char':
                return obj[0].encode('utf-16-le')
            if targetType == 'Byte':
                return bytes([obj])
            if targetType == 'Int16':
                return struct.pack('h', obj)
            if targetType == 'Int32':
                return struct.pack('i', obj)
            if targetType == 'Int64':
                return struct.pack('q', obj)
            if targetType == 'Single':
                return struct.pack('f', obj)
            if targetType == 'Double':
                return struct.pack('d', obj)
            if targetType == 'String':
                encodedMessage = obj.encode('utf-16-le')
                bytesToWrite = cls.toBytes(len(encodedMessage), 'Int32')
                bytesToWrite += encodedMessage
                return bytesToWrite

