from django.contrib import admin 
from .models import Movie, Rating, Recommendation, UserFeedback, Cinema, CinemaRating

class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'release_date', 'poster_path')
    search_fields = ('title',)
    list_filter = ('release_date',)

class RatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'score', 'date_rated')
    search_fields = ('user__username', 'movie__title')
    list_filter = ('score', 'date_rated')

class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'similar_movies_preview')
    search_fields = ('user__username', 'movie__title')
    list_filter = ('user',)
    ordering = ('user',)
    list_per_page = 20  # Customize the number of items displayed per page

    def similar_movies_preview(self, obj):
        if obj.similar_movies:
            return ', '.join(movie.get('title', 'Unknown') for movie in obj.similar_movies[:3])
        return "No similar movies"
    similar_movies_preview.short_description = 'Similar Movies Preview'

class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'recommendation', 'feedback_text', 'created_at')
    search_fields = ('user__username', 'recommendation__movie__title', 'feedback_text')
    list_filter = ('created_at', 'rating')  # Utilisez created_at pour filtrer
    ordering = ('created_at',)  # Trier par created_at
    list_per_page = 20  # Customize the number of items displayed per page
    
class CinemaAdmin(admin.ModelAdmin):
    # Ensure all fields used in list_display exist in the Cinema model
    list_display = ('name', 'description', 'location', 'date_created')
    search_fields = ('name', 'location')  # Fields that can be searched in the admin interface
    list_filter = ('date_created',)  # Fields used to filter the list in the admin
    ordering = ('-date_created',)  # Use descending order to show most recent first
    list_per_page = 20  # Number of items displayed per page in the admin interface
    
    
class CinemaRatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'cinema', 'score', 'date_rated')  # Display fields in the admin list
    search_fields = ('user__username', 'cinema__name', 'review')  # Search by user, cinema name, and review text
    list_filter = ('score', 'date_rated')  # Filter by score and date
    ordering = ('-date_rated',)  # Order by date rated, descending
    list_per_page = 20  # Customize the number of items displayed per page
    
# Register your models here.
admin.site.register(Movie, MovieAdmin)
admin.site.register(Rating, RatingAdmin)
admin.site.register(Recommendation, RecommendationAdmin)
admin.site.register(UserFeedback, UserFeedbackAdmin)  # Register UserFeedback with the admin
admin.site.register(Cinema, CinemaAdmin) 
admin.site.register(CinemaRating, CinemaRatingAdmin)
