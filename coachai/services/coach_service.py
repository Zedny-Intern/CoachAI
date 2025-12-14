"""Service layer for CoachAI."""

from coachai.services.coach_service_base import CoachServiceBase
from coachai.services.coach_service_helpers import CoachServiceHelpersMixin
from coachai.services.coach_service_generation import CoachServiceGenerationMixin
from coachai.services.coach_service_persistence import CoachServicePersistenceMixin


class CoachService(
    CoachServiceBase,
    CoachServiceHelpersMixin,
    CoachServiceGenerationMixin,
    CoachServicePersistenceMixin,
):
    pass