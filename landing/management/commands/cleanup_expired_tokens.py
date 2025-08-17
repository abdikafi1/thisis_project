from django.core.management.base import BaseCommand
from django.utils import timezone
from landing.models import PasswordResetToken


class Command(BaseCommand):
    help = 'Clean up expired password reset tokens from the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        # Find expired tokens
        expired_tokens = PasswordResetToken.objects.filter(
            expires_at__lt=timezone.now()
        )
        
        expired_count = expired_tokens.count()
        
        if expired_count == 0:
            self.stdout.write(
                self.style.SUCCESS('✅ No expired tokens found')
            )
            return
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f'🔍 Found {expired_count} expired tokens (dry run - no deletion)'
                )
            )
            
            for token in expired_tokens[:10]:  # Show first 10
                self.stdout.write(
                    f'   - {token.user.username}: expires {token.expires_at}'
                )
            
            if expired_count > 10:
                self.stdout.write(f'   ... and {expired_count - 10} more')
        else:
            # Actually delete expired tokens
            deleted_count = expired_tokens.delete()[0]
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'🧹 Successfully cleaned up {deleted_count} expired tokens'
                )
            )

