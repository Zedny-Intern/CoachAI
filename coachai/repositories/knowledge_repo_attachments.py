from typing import Dict, Any, Optional
import uuid


class KnowledgeRepositoryAttachmentsMixin:
    def upload_attachment(self, owner_id: str, bucket: str, path: str, file_bytes: bytes, content_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        bucket = self._user_bucket(owner_id)

        if not path:
            ext = 'bin'
            ct = (content_type or '').lower()
            if 'png' in ct:
                ext = 'png'
            elif 'jpeg' in ct or 'jpg' in ct:
                ext = 'jpg'
            elif 'webp' in ct:
                ext = 'webp'
            path = f"attachments/{uuid.uuid4().hex}.{ext}"

        svc = self._get_supabase_service()
        if not svc:
            self._log('upload_attachment: service client not available (missing SUPABASE_SERVICE_ROLE_KEY?)')
            return None
        try:
            svc.storage_create_bucket(bucket, public=False)
        except Exception as e:
            self._log(f'upload_attachment: create_bucket warning: {repr(e)}')

        sup = self._get_supabase()
        if not sup:
            self._log('upload_attachment: user client not available')
            return None
        try:
            svc.storage_upload(bucket, path, file_bytes, content_type=content_type)

            signed = ''
            try:
                signed_res = svc.storage_create_signed_url(bucket, path, expires_in=3600)
                if isinstance(signed_res, dict):
                    signed = signed_res.get('signedURL') or signed_res.get('signedUrl') or signed_res.get('signed_url') or ''
                else:
                    signed = getattr(signed_res, 'signedURL', None) or getattr(signed_res, 'signedUrl', None) or ''
            except Exception as e:
                self._log(f'upload_attachment: create_signed_url failed: {repr(e)}')

            rec = {'owner_id': owner_id, 'bucket': bucket, 'path': path, 'public_url': signed}
            res = sup.table_insert('attachments', rec)
            return res.data[0] if getattr(res, 'data', None) else None
        except Exception as e:
            self._log(f'upload_attachment failed: {repr(e)} bucket={bucket} path={path}')
            return None
