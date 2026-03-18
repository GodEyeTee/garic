"""Base adapter — abstract interface สำหรับแปลงข้อมูลเป็น StandardFeatureVector.

ทั้ง HistoricalAdapter และ LiveAdapter ต้อง implement interface นี้
เพื่อให้ได้ StandardFeatureVector ที่เหมือนกัน 100%.
"""

from abc import ABC, abstractmethod
from data.schema import StandardFeatureVector


class BaseDataAdapter(ABC):
    """Abstract interface สำหรับ data source ทุกประเภท."""

    @abstractmethod
    def get_feature_vector(self, symbol: str, timestamp: int) -> StandardFeatureVector:
        """สร้าง StandardFeatureVector ณ เวลาที่กำหนด.

        *** ต้อง return feature vector ที่มี shape และ meaning เหมือนกัน
            ไม่ว่าจะเป็น historical หรือ live ***
        """
        ...

    @abstractmethod
    def is_candle_closed(self) -> bool:
        """ตรวจสอบว่าแท่งเทียนปิดแล้วหรือยัง.

        *** สำคัญมาก: โมเดลจะ action เฉพาะเมื่อ return True ***
        - Historical: always True (ทุก row คือ candle ที่ปิดแล้ว)
        - Live: True เฉพาะเมื่อ candle close event เกิดขึ้น
        """
        ...
